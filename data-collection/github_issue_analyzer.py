#!/usr/bin/env python3
"""
GitHub Issue Analyzer
Analyzes GitHub issues, finds associated PRs in comments, and extracts fix details.
"""

import requests
import re
import sys
import json
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
import time

class GitHubIssueAnalyzer:
    def __init__(self, github_token: Optional[str] = None):
        """
        Initialize the analyzer with optional GitHub token for higher rate limits.
        
        Args:
            github_token: GitHub personal access token (optional)
        """
        self.session = requests.Session()
        self.headers = {
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'GitHub-Issue-Analyzer'
        }
        if github_token:
            self.headers['Authorization'] = f'token {github_token}'
        self.session.headers.update(self.headers)
        
    def parse_github_url(self, url: str) -> Tuple[str, str, int]:
        """
        Parse GitHub issue URL to extract owner, repo, and issue number.
        
        Args:
            url: GitHub issue URL
            
        Returns:
            Tuple of (owner, repo, issue_number)
        """
        # Handle various GitHub URL formats
        patterns = [
            r'github\.com/([^/]+)/([^/]+)/issues/(\d+)',
            r'api\.github\.com/repos/([^/]+)/([^/]+)/issues/(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1), match.group(2), int(match.group(3))
        
        raise ValueError(f"Invalid GitHub issue URL: {url}")
    
    def fetch_issue_details(self, owner: str, repo: str, issue_num: int) -> Dict:
        """
        Fetch issue details from GitHub API.
        
        Args:
            owner: Repository owner
            repo: Repository name
            issue_num: Issue number
            
        Returns:
            Issue details as dictionary
        """
        api_url = f'https://api.github.com/repos/{owner}/{repo}/issues/{issue_num}'
        
        try:
            response = self.session.get(api_url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to fetch issue details: {e}")
    
    def fetch_issue_comments(self, owner: str, repo: str, issue_num: int) -> List[Dict]:
        """
        Fetch all comments for an issue.
        
        Args:
            owner: Repository owner
            repo: Repository name
            issue_num: Issue number
            
        Returns:
            List of comment dictionaries
        """
        api_url = f'https://api.github.com/repos/{owner}/{repo}/issues/{issue_num}/comments'
        comments = []
        page = 1
        
        while True:
            try:
                response = self.session.get(api_url, params={'page': page, 'per_page': 100})
                response.raise_for_status()
                page_comments = response.json()
                
                if not page_comments:
                    break
                    
                comments.extend(page_comments)
                page += 1
                
                # Respect rate limiting
                if 'X-RateLimit-Remaining' in response.headers:
                    if int(response.headers['X-RateLimit-Remaining']) < 10:
                        time.sleep(2)
                        
            except requests.exceptions.RequestException as e:
                raise Exception(f"Failed to fetch comments: {e}")
        
        return comments
    
    def fetch_issue_events(self, owner: str, repo: str, issue_num: int) -> List[Dict]:
        """
        Fetch issue events (including close events with PR references).
        
        Args:
            owner: Repository owner
            repo: Repository name
            issue_num: Issue number
            
        Returns:
            List of event dictionaries
        """
        api_url = f'https://api.github.com/repos/{owner}/{repo}/issues/{issue_num}/events'
        events = []
        page = 1
        
        while True:
            try:
                response = self.session.get(api_url, params={'page': page, 'per_page': 100})
                response.raise_for_status()
                page_events = response.json()
                
                if not page_events:
                    break
                    
                events.extend(page_events)
                page += 1
                
                # Respect rate limiting
                if 'X-RateLimit-Remaining' in response.headers:
                    if int(response.headers['X-RateLimit-Remaining']) < 10:
                        time.sleep(2)
                        
            except requests.exceptions.RequestException as e:
                break
        
        return events
    
    def fetch_issue_timeline(self, owner: str, repo: str, issue_num: int) -> List[Dict]:
        """
        Fetch issue timeline events (better for finding linked PRs).
        
        Args:
            owner: Repository owner
            repo: Repository name
            issue_num: Issue number
            
        Returns:
            List of timeline event dictionaries
        """
        api_url = f'https://api.github.com/repos/{owner}/{repo}/issues/{issue_num}/timeline'
        timeline = []
        page = 1
        
        # Timeline API requires special accept header
        old_headers = self.session.headers.copy()
        self.session.headers['Accept'] = 'application/vnd.github.mockingbird-preview+json'
        
        try:
            while True:
                try:
                    response = self.session.get(api_url, params={'page': page, 'per_page': 100})
                    response.raise_for_status()
                    page_timeline = response.json()
                    
                    if not page_timeline:
                        break
                        
                    timeline.extend(page_timeline)
                    page += 1
                    
                    # Respect rate limiting
                    if 'X-RateLimit-Remaining' in response.headers:
                        if int(response.headers['X-RateLimit-Remaining']) < 10:
                            time.sleep(2)
                            
                except requests.exceptions.RequestException as e:
                    break
        finally:
            # Restore original headers
            self.session.headers = old_headers
        
        return timeline
    
    def find_pr_references(self, text: str) -> List[Dict]:
        """
        Find PR references in text (comments, issue body).
        
        Args:
            text: Text to search for PR references
            
        Returns:
            List of PR references with details
        """
        pr_refs = []
        
        # Patterns to match PR references
        patterns = [
            # Direct PR URLs
            (r'https://github\.com/([^/]+)/([^/]+)/pull/(\d+)', 'url'),
            # "closed as completed in #123" or "closed this in #123"
            (r'closed\s+(?:as\s+completed\s+)?(?:this\s+)?in\s*#(\d+)', 'closed_in'),
            # "Fixed in PR #123" or similar
            (r'(?:fixed|resolved|closed|merged)\s+(?:in|by|with)?\s*(?:PR|pr)?\s*#(\d+)', 'fix_mention'),
            # "PR #123" or "Pull Request #123"
            (r'(?:PR|pr|Pull [Rr]equest)\s*#(\d+)', 'pr_mention'),
            # PR mentions like #123 (but more selective to avoid false positives)
            (r'(?:^|\s)#(\d{3,6})(?:\s|$)', 'mention'),
        ]
        
        for pattern, ref_type in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if ref_type == 'url':
                    pr_refs.append({
                        'type': ref_type,
                        'owner': match.group(1),
                        'repo': match.group(2),
                        'number': int(match.group(3)),
                        'match': match.group(0)
                    })
                else:
                    # For mentions, we'll need to get owner/repo from context
                    pr_refs.append({
                        'type': ref_type,
                        'number': int(match.group(1)),
                        'match': match.group(0)
                    })
        
        return pr_refs
    
    def fetch_pr_details(self, owner: str, repo: str, pr_num: int) -> Dict:
        """
        Fetch PR details from GitHub API.
        
        Args:
            owner: Repository owner
            repo: Repository name
            pr_num: PR number
            
        Returns:
            PR details as dictionary
        """
        api_url = f'https://api.github.com/repos/{owner}/{repo}/pulls/{pr_num}'
        
        try:
            response = self.session.get(api_url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return None
    
    def fetch_pr_files(self, owner: str, repo: str, pr_num: int) -> List[Dict]:
        """
        Fetch changed files in a PR.
        
        Args:
            owner: Repository owner
            repo: Repository name
            pr_num: PR number
            
        Returns:
            List of changed files
        """
        api_url = f'https://api.github.com/repos/{owner}/{repo}/pulls/{pr_num}/files'
        
        try:
            response = self.session.get(api_url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return []
    
    def extract_fix_summary(self, pr_data: Dict, pr_files: List[Dict]) -> str:
        """
        Extract a summary of the fix from PR data.
        
        Args:
            pr_data: PR details
            pr_files: List of changed files
            
        Returns:
            Fix summary as string
        """
        summary_parts = []
        
        # PR title and description
        if pr_data:
            summary_parts.append(f"PR Title: {pr_data.get('title', 'N/A')}")
            
            # Extract key information from PR body
            body = pr_data.get('body', '')
            if body:
                # Look for common sections in PR descriptions
                sections = ['fix', 'solution', 'changes', 'what', 'why']
                for section in sections:
                    pattern = rf'(?:^|\n)#+?\s*{section}.*?:(.*?)(?=\n#|\Z)'
                    match = re.search(pattern, body, re.IGNORECASE | re.DOTALL)
                    if match:
                        content = match.group(1).strip()[:500]  # Limit length
                        if content:
                            summary_parts.append(f"{section.capitalize()}: {content}")
            
            # Add merge status
            if pr_data.get('merged'):
                summary_parts.append(f"Status: Merged on {pr_data.get('merged_at', 'N/A')}")
            else:
                summary_parts.append(f"Status: {pr_data.get('state', 'Unknown')}")
        
        # Summary of changed files
        if pr_files:
            file_summary = []
            for file in pr_files[:5]:  # Limit to first 5 files
                filename = file.get('filename', 'Unknown')
                additions = file.get('additions', 0)
                deletions = file.get('deletions', 0)
                file_summary.append(f"  - {filename} (+{additions}/-{deletions})")
            
            if file_summary:
                summary_parts.append("Changed Files:\n" + "\n".join(file_summary))
                if len(pr_files) > 5:
                    summary_parts.append(f"  ... and {len(pr_files) - 5} more files")
        
        return "\n".join(summary_parts)
    
    def analyze_issue(self, issue_url: str) -> Dict:
        """
        Main method to analyze an issue and find its fix.
        
        Args:
            issue_url: GitHub issue URL
            
        Returns:
            Dictionary with issue and fix details
        """
        # Parse the URL
        owner, repo, issue_num = self.parse_github_url(issue_url)
        
        # Fetch issue details
        issue_data = self.fetch_issue_details(owner, repo, issue_num)
        
        # Fetch comments
        comments = self.fetch_issue_comments(owner, repo, issue_num)
        
        # Fetch events to find closing PRs
        events = self.fetch_issue_events(owner, repo, issue_num)
        
        # Fetch timeline for better PR tracking
        timeline = self.fetch_issue_timeline(owner, repo, issue_num)
        
        # Find PR references in issue body and comments
        all_pr_refs = []
        
        # Check issue body
        if issue_data.get('body'):
            all_pr_refs.extend(self.find_pr_references(issue_data['body']))
        
        # Check comments
        for comment in comments:
            if comment.get('body'):
                refs = self.find_pr_references(comment['body'])
                for ref in refs:
                    # Add context about where the reference was found
                    ref['found_in'] = f"Comment by {comment.get('user', {}).get('login', 'Unknown')}"
                    ref['comment_date'] = comment.get('created_at', 'Unknown')
                all_pr_refs.extend(refs)
        
        # Check timeline for PR references
        for event in timeline:
            event_type = event.get('event')
            
            # Look for closed_event with PR source
            if event_type == 'closed':
                actor = event.get('actor', {})
                commit_id = event.get('commit_id')
                commit_url = event.get('commit_url')
                
                # Sometimes the closing PR info is in the commit URL
                if commit_url:
                    # Try to fetch the commit to see if it's part of a PR
                    pr_match = re.search(r'/pull/(\d+)', str(commit_url))
                    if pr_match:
                        all_pr_refs.append({
                            'type': 'closing_commit',
                            'owner': owner,
                            'repo': repo,
                            'number': int(pr_match.group(1)),
                            'found_in': 'Closing commit',
                            'match': f"Issue closed by PR #{pr_match.group(1)}"
                        })
            
            # Look for cross-referenced events with PR sources
            elif event_type == 'cross-referenced':
                source = event.get('source', {})
                if source.get('type') == 'issue':
                    issue_source = source.get('issue', {})
                    # Check if this is actually a PR
                    if 'pull_request' in issue_source and issue_source.get('pull_request'):
                        pr_num = issue_source.get('number')
                        if pr_num:
                            all_pr_refs.append({
                                'type': 'cross_reference',
                                'owner': owner,
                                'repo': repo,
                                'number': pr_num,
                                'found_in': 'Cross-referenced PR',
                                'match': f"Referenced in PR #{pr_num}"
                            })
            
            # Look for mentioned events in PRs
            elif event_type == 'mentioned':
                all_pr_refs.extend(self.find_pr_references(event.get('body', '')))
        
        # Check events for closing PRs (fallback for older API)
        for event in events:
            if event.get('event') in ['closed', 'merged', 'referenced']:
                # Check commit message for PR references
                if event.get('commit_id'):
                    commit_url = event.get('commit_url')
                    if commit_url:
                        # Extract PR number from commit URL if it contains one
                        pr_match = re.search(r'/pull/(\d+)', str(commit_url))
                        if pr_match:
                            all_pr_refs.append({
                                'type': 'event_close',
                                'owner': owner,
                                'repo': repo,
                                'number': int(pr_match.group(1)),
                                'found_in': 'Issue close event',
                                'match': f"Closed by PR #{pr_match.group(1)}"
                            })
                
                # Check for cross-referenced events
                if event.get('event') == 'cross-referenced':
                    source = event.get('source', {})
                    if source.get('type') == 'issue' and source.get('issue'):
                        issue_ref = source['issue']
                        # Check if this is actually a PR (GitHub uses same endpoint for PRs)
                        if 'pull_request' in issue_ref:
                            pr_num = issue_ref.get('number')
                            if pr_num:
                                all_pr_refs.append({
                                    'type': 'cross_reference',
                                    'owner': owner,
                                    'repo': repo,
                                    'number': pr_num,
                                    'found_in': 'Cross-reference event',
                                    'match': f"Cross-referenced in PR #{pr_num}"
                                })
        
        # Deduplicate PR references
        seen_prs = set()
        unique_pr_refs = []
        for ref in all_pr_refs:
            # For mentions without owner/repo, use the issue's owner/repo
            if 'owner' not in ref:
                ref['owner'] = owner
                ref['repo'] = repo
            
            pr_key = (ref.get('owner'), ref.get('repo'), ref.get('number'))
            if pr_key not in seen_prs and pr_key[2] is not None:
                seen_prs.add(pr_key)
                unique_pr_refs.append(ref)
        
        # Fetch PR details for each reference
        pr_details = []
        for ref in unique_pr_refs:
            pr_data = self.fetch_pr_details(ref['owner'], ref['repo'], ref['number'])
            if pr_data:
                pr_files = self.fetch_pr_files(ref['owner'], ref['repo'], ref['number'])
                pr_details.append({
                    'reference': ref,
                    'data': pr_data,
                    'files': pr_files,
                    'summary': self.extract_fix_summary(pr_data, pr_files)
                })
        
        # Compile results
        result = {
            'issue': {
                'url': issue_url,
                'number': issue_num,
                'title': issue_data.get('title', 'N/A'),
                'state': issue_data.get('state', 'Unknown'),
                'created_at': issue_data.get('created_at', 'Unknown'),
                'body': issue_data.get('body', '')[:1000] if issue_data.get('body') else 'No description'
            },
            'prs': pr_details,
            'summary': self.generate_final_summary(issue_data, pr_details)
        }
        
        return result
    
    def generate_final_summary(self, issue_data: Dict, pr_details: List[Dict]) -> str:
        """
        Generate a simple summary with just issue and fix descriptions.
        
        Args:
            issue_data: Issue details
            pr_details: List of PR details
            
        Returns:
            Plain text with issue and fix descriptions
        """
        output = []
        
        # Issue Description
        output.append("ISSUE DESCRIPTION:")
        body = issue_data.get('body', '').strip()
        if body:
            # Clean up the description - remove markdown formatting
            body = re.sub(r'```[^`]*```', '', body)  # Remove code blocks
            body = re.sub(r'`[^`]+`', '', body)  # Remove inline code
            body = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', body)  # Convert links to text
            body = re.sub(r'[#*_~]', '', body)  # Remove markdown symbols
            body = re.sub(r'\n+', ' ', body)  # Replace multiple newlines with space
            body = re.sub(r'\s+', ' ', body).strip()  # Normalize whitespace
            output.append(body[:1000])  # Limit to 1000 chars
        else:
            output.append("No description provided")
        
        output.append("\n")
        
        # Fix Description
        output.append("FIX DESCRIPTION:")
        
        if pr_details:
            # Focus on merged PRs first, then open PRs
            merged_prs = [pr for pr in pr_details if pr['data'].get('merged')]
            relevant_prs = merged_prs if merged_prs else pr_details
            
            if relevant_prs:
                # Get the most relevant PR (first merged PR or first PR)
                main_pr = relevant_prs[0]
                pr_body = main_pr['data'].get('body', '').strip()
                
                if pr_body:
                    # Clean up PR body
                    pr_body = re.sub(r'```[^`]*```', '', pr_body)  # Remove code blocks
                    pr_body = re.sub(r'`[^`]+`', '', pr_body)  # Remove inline code
                    pr_body = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', pr_body)  # Convert links to text
                    pr_body = re.sub(r'[#*_~]', '', pr_body)  # Remove markdown symbols
                    pr_body = re.sub(r'\n+', ' ', pr_body)  # Replace multiple newlines with space
                    pr_body = re.sub(r'\s+', ' ', pr_body).strip()  # Normalize whitespace
                    
                    # Try to extract the actual fix description from common PR template sections
                    fix_patterns = [
                        r'(?:fix|solution|changes?|what)[:|\n](.*?)(?:why|test|how|$)',
                        r'(?:description)[:|\n](.*?)(?:test|checklist|$)',
                        r'(?:summary)[:|\n](.*?)(?:test|details|$)'
                    ]
                    
                    fix_text = None
                    for pattern in fix_patterns:
                        match = re.search(pattern, pr_body, re.IGNORECASE | re.DOTALL)
                        if match:
                            fix_text = match.group(1).strip()
                            if fix_text:
                                break
                    
                    if fix_text:
                        output.append(fix_text[:1000])
                    else:
                        # Fall back to PR title and first part of body
                        pr_title = main_pr['data'].get('title', '')
                        combined = f"{pr_title}. {pr_body[:500]}" if pr_title else pr_body[:1000]
                        output.append(combined)
                else:
                    # Use PR title if no body
                    output.append(main_pr['data'].get('title', 'No fix description available'))
            else:
                output.append("No fix description found")
        else:
            output.append("No pull requests found - issue may be unresolved or fixed without a PR")
        
        return "\n".join(output)


def main():
    """Main function to run the analyzer."""
    if len(sys.argv) < 2:
        print("Usage: python github_issue_analyzer.py <github_issue_url> [github_token]")
        print("\nExample:")
        print("  python github_issue_analyzer.py https://github.com/owner/repo/issues/123")
        print("  python github_issue_analyzer.py https://github.com/owner/repo/issues/123 ghp_YOUR_TOKEN")
        print("\nNote: Providing a GitHub token increases API rate limits.")
        sys.exit(1)
    
    issue_url = sys.argv[1]
    github_token = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        # Initialize analyzer
        analyzer = GitHubIssueAnalyzer(github_token)
        
        # Analyze the issue
        result = analyzer.analyze_issue(issue_url)
        
        # Print just the issue and fix descriptions
        print(result['summary'])
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()