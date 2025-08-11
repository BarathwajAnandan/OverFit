/* global ForceGraph3D */

// Load real graph data from KuzuDB export
let placeholderGraph = {
  nodes: [],
  links: []
}

// Load real data from exported JSON
async function loadGraphData()
{
  try
  {
    const response = await fetch('./assets/graph_data.json')
    const data = await response.json()
    placeholderGraph = {
      nodes: data.nodes,
      links: data.links
    }
    console.log(`Loaded ${data.node_count} nodes and ${data.link_count} links from KuzuDB`)
    
    // Reinitialize graph with real data
    if (window.__Graph)
    {
      window.__Graph.graphData(placeholderGraph)
    }
    renderList()
  }
  catch (error)
  {
    console.error('Failed to load graph data:', error)
    // Fallback to sample data if needed
    placeholderGraph = {
      nodes: [
        { id: 'p1', type: 'problem', label: 'Sample Problem', karma: 5, views: 100 },
        { id: 's1', type: 'solution', label: 'Sample Solution', reuseScore: 0.8, upvotes: 10 },
        { id: 'm1', type: 'model', label: 'Sample Model', size: 'Large', provider: 'Test' }
      ],
      links: [
        { source: 'p1', target: 's1', type: 'solves', strength: 1.0 }
      ]
    }
    if (window.__Graph)
    {
      window.__Graph.graphData(placeholderGraph)
    }
    renderList()
  }
}

const state = {
  showLabels: false,
  pinnedNodes: new Set(),
  activeTab: 'problems',
  sortBy: 'relevance',
  karma: Number(window.localStorage.getItem('karmaScore') || 42),
  seed: Number(window.localStorage.getItem('seedScore') || 2),
  leech: Number(window.localStorage.getItem('leechScore') || 0)
}

function getLife()
{
  return state.seed - state.leech
}

function updateScoresUI()
{
  const seedEl = document.getElementById('seedScore')
  const leechEl = document.getElementById('leechScore')
  const lifeEl = document.getElementById('lifeScore')
  if (seedEl)
  {
    seedEl.textContent = String(state.seed)
  }
  if (leechEl)
  {
    leechEl.textContent = String(state.leech)
  }
  if (lifeEl)
  {
    const life = getLife()
    lifeEl.textContent = String(life)
    const lifeWrap = lifeEl.parentElement
    if (lifeWrap)
    {
      lifeWrap.classList.remove('positive', 'negative')
      lifeWrap.classList.add(life >= 0 ? 'positive' : 'negative')
    }
  }
  window.localStorage.setItem('seedScore', String(state.seed))
  window.localStorage.setItem('leechScore', String(state.leech))
  window.localStorage.setItem('karmaScore', String(state.karma))
}

function nodeColor(node)
{
  if (node.type === 'problem')
  {
    return '#fdba74'
  }
  if (node.type === 'solution')
  {
    return '#34d399'
  }
  if (node.type === 'model')
  {
    return '#60a5fa'
  }
  if (node.type === 'concept')
  {
    return '#a78bfa'
  }
  if (node.type === 'method')
  {
    return '#fb7185'
  }
  return '#9ca3af'
}

function linkColor(link)
{
  if (link.type === 'similarity' || link.type === 'similar_to')
  {
    return 'rgba(122,162,255,0.8)'
  }
  if (link.type === 'solves')
  {
    return 'rgba(52,211,153,0.9)'
  }
  if (link.type === 'produced_by' || link.type === 'can_solve')
  {
    return 'rgba(96,165,250,0.9)'
  }
  if (link.type === 'about' || link.type === 'uses' || link.type === 'references')
  {
    return 'rgba(167,139,250,0.7)'
  }
  return 'rgba(234,179,8,0.9)'
}

function initGraph()
{
  const elem = document.getElementById('graph')
  const Graph = ForceGraph3D()(elem)
    .graphData(placeholderGraph)
    .nodeAutoColorBy('type')
    .nodeVal(node => {
      if (node.type === 'model') return 12
      if (node.type === 'solution') return 8  
      if (node.type === 'problem') return 10
      if (node.type === 'concept') return 6
      if (node.type === 'method') return 7
      return 5
    })
    .nodeColor(node => nodeColor(node))
    .nodeOpacity(0.95)
    .nodeThreeObjectExtend(true)
    .nodeLabel(node => state.showLabels ? node.label : '')
    .linkColor(link => linkColor(link))
    .linkOpacity(0.35)
    .linkCurvature(link => 0.35 + Math.random() * 0.3)
    .linkDirectionalParticles(2)
    .linkDirectionalParticleWidth(link => Math.max(1, (link.strength || 0.2) * 4))
    .linkDirectionalParticleSpeed(0.005)
    .backgroundColor('#0b0d12')
    .onNodeClick(node => onNodeClick(node, Graph))
    .onBackgroundClick(() => closeDetails())

  Graph.cameraPosition({ x: 0, y: 0, z: 180 }, { x: 0, y: 0, z: 0 }, 1500)

  window.__Graph = Graph
}

function onNodeClick(node, Graph)
{
  const distance = 80
  const distRatio = 1 + distance / Math.hypot(node.x || 1, node.y || 1, node.z || 1)
  const newPos = { x: (node.x || 1) * distRatio, y: (node.y || 1) * distRatio, z: (node.z || 1) * distRatio }
  Graph.cameraPosition(newPos, node, 800)

  openDetails(node)
}

function openDetails(node)
{
  const panel = document.getElementById('detailsPanel')
  const content = document.getElementById('detailsContent')
  const title = document.getElementById('detailsTitle')
  const typeEl = document.getElementById('detailsType')
  const graphSection = document.querySelector('.graph-section')

  // Set title and type
  title.textContent = node.label || node.name || 'Node Details'
  typeEl.textContent = node.type || 'unknown'
  typeEl.className = `details-type ${node.type || 'unknown'}`

  // Define the properties to show and their display configuration
  const propertiesToShow = [
    { key: 'id', label: 'ID', type: 'text' },
    { key: 'type', label: 'Type', type: 'text' },
    { key: 'name', label: 'Model Name', type: 'text', condition: node => node.type === 'model' },
    { key: 'label', label: 'Label', type: 'text' },
    { key: 'text', label: 'Text', type: 'text' },
    { key: 'problem_type', label: 'Problem Type', type: 'text', condition: node => node.type === 'problem' },
    { key: 'variables', label: 'Variables', type: 'code' },
    { key: 'metadata', label: 'Metadata', type: 'code' }
  ]

  // Build the content HTML
  const sections = []
  
  propertiesToShow.forEach(prop => {
    // Skip if there's a condition and it's not met
    if (prop.condition && !prop.condition(node)) {
      return
    }
    
    const value = node[prop.key]
    
    // Skip if value doesn't exist or is empty
    if (value === undefined || value === null || value === '') {
      return
    }
    
    let displayValue = value
    let valueClass = 'details-value'
    
    // Format the value based on type
    if (prop.type === 'code') {
      valueClass += ' code'
      if (typeof value === 'object') {
        displayValue = JSON.stringify(value, null, 2)
      }
    } else {
      displayValue = String(value)
    }
    
    sections.push(`
      <div class="details-section">
        <label class="details-label">${prop.label}</label>
        <div class="${valueClass}">${displayValue}</div>
      </div>
    `)
  })
  
  // If no sections, show a message
  if (sections.length === 0) {
    content.innerHTML = '<div class="details-section"><div class="details-value empty">No details available</div></div>'
  } else {
    content.innerHTML = sections.join('')
  }
  
  panel.classList.remove('hidden')
  if (graphSection) {
    graphSection.classList.add('sidebar-open')
  }
}

function closeDetails()
{
  const panel = document.getElementById('detailsPanel')
  const graphSection = document.querySelector('.graph-section')
  
  panel.classList.add('hidden')
  if (graphSection) {
    graphSection.classList.remove('sidebar-open')
  }
}



function initHeader()
{
  updateScoresUI()

  const labelsBtn = document.getElementById('toggleLabels')
  if (labelsBtn)
  {
    labelsBtn.addEventListener('click', () =>
    {
      state.showLabels = !state.showLabels
      if (window.__Graph)
      {
        window.__Graph.nodeLabel(node => state.showLabels ? node.label : '')
      }
    })
  }

  const resetBtn = document.getElementById('resetCamera')
  if (resetBtn)
  {
    resetBtn.addEventListener('click', () =>
    {
      if (window.__Graph)
      {
        window.__Graph.cameraPosition({ x: 0, y: 0, z: 180 }, { x: 0, y: 0, z: 0 }, 1200)
      }
    })
  }

  const pinBtn = document.getElementById('pinSelected')
  if (pinBtn)
  {
    pinBtn.addEventListener('click', () =>
    {
      const panel = document.getElementById('detailsPanel')
      if (panel.classList.contains('hidden'))
      {
        return
      }
      const nodeLabel = document.getElementById('detailsTitle').textContent
      const node = placeholderGraph.nodes.find(n => n.label === nodeLabel)
      if (!node)
      {
        return
      }
      if (state.pinnedNodes.has(node.id))
      {
        node.fx = undefined; node.fy = undefined; node.fz = undefined
        state.pinnedNodes.delete(node.id)
      }
      else
      {
        node.fx = node.x; node.fy = node.y; node.fz = node.z
        state.pinnedNodes.add(node.id)
      }
    })
  }

  const globalSearch = document.getElementById('globalSearch')
  if (globalSearch)
  {
    globalSearch.addEventListener('keydown', e =>
    {
      if (e.key === 'Enter')
      {
        e.preventDefault()
        const q = e.currentTarget.value.trim().toLowerCase()
        renderList(q)
      }
    })
  }

  const contribBtn = document.getElementById('contributeCta')
  if (contribBtn)
  {
    contribBtn.addEventListener('click', () =>
    {
      state.activeTab = 'unanswered'
      document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === 'unanswered'))
      renderList()
    })
  }

  const filterDomain = document.getElementById('filterDomain')
  if (filterDomain)
  {
    filterDomain.addEventListener('change', () => renderList())
  }
  const filterModels = document.getElementById('filterModels')
  if (filterModels)
  {
    filterModels.addEventListener('change', () => renderList())
  }
}

function toast(message)
{
  let holder = document.getElementById('toastHolder')
  if (!holder)
  {
    holder = document.createElement('div')
    holder.id = 'toastHolder'
    holder.style.position = 'fixed'
    holder.style.right = '20px'
    holder.style.bottom = '100px'
    holder.style.display = 'grid'
    holder.style.gap = '8px'
    holder.style.zIndex = '100'
    document.body.appendChild(holder)
  }
  const t = document.createElement('div')
  t.textContent = message
  t.style.padding = '10px 12px'
  t.style.background = 'rgba(20,24,36,0.9)'
  t.style.border = '1px solid var(--border)'
  t.style.borderRadius = '10px'
  t.style.boxShadow = 'var(--shadow)'
  holder.appendChild(t)

  const a11y = document.getElementById('a11yToasts')
  if (a11y)
  {
    a11y.textContent = message
  }

  setTimeout(() =>
  {
    t.remove()
  }, 2000)
}

function initTabs()
{
  const tabs = document.querySelectorAll('.tab')
  tabs.forEach(tab =>
  {
    tab.addEventListener('click', () =>
    {
      tabs.forEach(t => t.classList.remove('active'))
      tab.classList.add('active')
      state.activeTab = tab.dataset.tab
      renderList()
    })
  })
}

function initList()
{
  const sortSelect = document.getElementById('sortSelect')
  if (sortSelect)
  {
    sortSelect.addEventListener('change', e =>
    {
      state.sortBy = e.target.value
      renderList()
    })
  }
  renderList()
}

function getItemsForTab()
{
  if (state.activeTab === 'problems')
  {
    return placeholderGraph.nodes.filter(n => n.type === 'problem')
  }
  if (state.activeTab === 'unanswered')
  {
    return placeholderGraph.nodes.filter(n => n.type === 'problem' && countAnswers(n.id) === 0)
  }
  if (state.activeTab === 'solutions')
  {
    return placeholderGraph.nodes.filter(n => n.type === 'solution')
  }
  return placeholderGraph.nodes.filter(n => n.type === 'model')
}

function sortItems(items)
{
  switch (state.sortBy)
  {
    case 'reuse':
      return items.slice().sort((a, b) => (b.reuseScore || 0) - (a.reuseScore || 0))
    case 'karma':
      return items.slice().sort((a, b) => (b.karma || 0) - (a.karma || 0))
    case 'cost':
      return items.slice().sort((a, b) => costRank(a.cost) - costRank(b.cost))
    case 'freshness':
      return items.slice().sort(() => 0.5 - Math.random())
    default:
      return items
  }
}

function costRank(cost)
{
  if (cost === '$')
  {
    return 1
  }
  if (cost === '$$')
  {
    return 2
  }
  if (cost === '$$$')
  {
    return 3
  }
  return 999
}

function countAnswers(problemId)
{
  const links = placeholderGraph.links.filter(l => l.type === 'solves' && (l.source === problemId || l.target === problemId))
  let count = 0
  links.forEach(l =>
  {
    const otherId = l.source === problemId ? l.target : l.source
    const other = placeholderGraph.nodes.find(n => n.id === otherId)
    if (other && other.type === 'solution')
    {
      count += 1
    }
  })
  return count
}

function itemMatchesFilters(item)
{
  const domain = document.getElementById('filterDomain')?.value || ''
  const model = document.getElementById('filterModels')?.value || ''

  if (domain && (item.domain || '') !== domain)
  {
    if (item.type !== 'solution')
    {
      return false
    }
  }

  if (!model)
  {
    return true
  }

  if (item.type === 'model')
  {
    return (item.label || '').toLowerCase() === model.toLowerCase()
  }

  if (item.type === 'solution')
  {
    const produced = placeholderGraph.links.find(l => l.type === 'produced_by' && (l.source === item.id || l.target === item.id))
    if (!produced)
    {
      return false
    }
    const modelId = produced.source === item.id ? produced.target : produced.source
    const modelNode = placeholderGraph.nodes.find(n => n.id === modelId)
    return modelNode && (modelNode.label || '').toLowerCase() === model.toLowerCase()
  }

  if (item.type === 'problem')
  {
    const solves = placeholderGraph.links.filter(l => l.type === 'solves' && (l.source === item.id || l.target === item.id))
    return solves.some(s =>
    {
      const solId = s.source === item.id ? s.target : s.source
      const produced = placeholderGraph.links.find(l => l.type === 'produced_by' && (l.source === solId || l.target === solId))
      if (!produced)
      {
        return false
      }
      const modelId = produced.source === solId ? produced.target : produced.source
      const modelNode = placeholderGraph.nodes.find(n => n.id === modelId)
      return modelNode && (modelNode.label || '').toLowerCase() === model.toLowerCase()
    })
  }
  return true
}

function renderList(query = '')
{
  const container = document.getElementById('listContainer')
  container.innerHTML = ''

  const items = sortItems(getItemsForTab())
  const filtered = items.filter(item =>
  {
    if (query)
    {
      const hay = `${item.label || ''} ${(item.tags || []).join(' ')} ${(item.domain || '')}`.toLowerCase()
      if (!hay.includes(query))
      {
        return false
      }
    }
    return itemMatchesFilters(item)
  })

  const table = document.createElement('table')
  table.className = 'table'

  const thead = document.createElement('thead')
  const thr = document.createElement('tr')

  if (state.activeTab === 'problems' || state.activeTab === 'unanswered')
  {
    thr.innerHTML = `<th style="width:90px;">Votes</th><th style="width:100px;">Answers</th><th style="width:90px;">Views</th><th>Title</th><th style="width:240px;">Tags</th><th style="width:220px;">Actions</th>`
  }
  else if (state.activeTab === 'solutions')
  {
    thr.innerHTML = `<th style="width:100px;">Upvotes</th><th style="width:100px;">Reuse</th><th style="width:100px;">Cost</th><th>Title</th><th style="width:180px;">Model</th><th style="width:180px;">Actions</th>`
  }
  else
  {
    thr.innerHTML = `<th>Model</th><th style="width:180px;">Provider</th><th style="width:120px;">Size</th><th style="width:160px;">Actions</th>`
  }

  thead.appendChild(thr)
  table.appendChild(thead)

  const tbody = document.createElement('tbody')

  filtered.forEach(item =>
  {
    const tr = document.createElement('tr')

    if (state.activeTab === 'problems' || state.activeTab === 'unanswered')
    {
      const votes = item.karma || 0
      const answers = countAnswers(item.id)
      const views = item.views || Math.floor(50 + Math.random() * 400)

      tr.innerHTML = `
        <td>${votes}</td>
        <td>${answers}</td>
        <td>${views}</td>
        <td>${item.label}</td>
        <td>${(item.tags || []).map(t => `<span class=\"badge\">${t}</span>`).join(' ')}</td>
        <td class="td-actions"></td>
      `

      const actions = tr.querySelector('.td-actions')
      const openBtn = document.createElement('button')
      openBtn.className = 'btn small'
      openBtn.textContent = 'Open'
      openBtn.addEventListener('click', () => openDetails(item))

      const answerBtn = document.createElement('button')
      answerBtn.className = 'btn small'
      answerBtn.textContent = answers === 0 ? 'Answer' : 'Add answer'
      answerBtn.addEventListener('click', () => onAnswerProblem(item))

      actions.appendChild(openBtn)
      actions.appendChild(answerBtn)
    }
    else if (state.activeTab === 'solutions')
    {
      const upvotes = item.upvotes || 0
      const reuse = item.reuseScore != null ? `${(item.reuseScore * 100).toFixed(0)}%` : '—'
      const produced = placeholderGraph.links.find(l => l.type === 'produced_by' && (l.source === item.id || l.target === item.id))
      const modelId = produced ? (produced.source === item.id ? produced.target : produced.source) : null
      const modelNode = modelId ? placeholderGraph.nodes.find(n => n.id === modelId) : null

      tr.innerHTML = `
        <td>${upvotes}</td>
        <td>${reuse}</td>
        <td>${item.cost || '—'}</td>
        <td>${item.label}</td>
        <td>${modelNode ? modelNode.label : '—'}</td>
        <td class="td-actions"></td>
      `

      const actions = tr.querySelector('.td-actions')
      const openBtn = document.createElement('button')
      openBtn.className = 'btn small'
      openBtn.textContent = 'Open'
      openBtn.addEventListener('click', () => openDetails(item))
      actions.appendChild(openBtn)
    }
    else
    {
      tr.innerHTML = `
        <td>${item.label}</td>
        <td>${item.provider || '—'}</td>
        <td>${item.size || '—'}</td>
        <td class="td-actions"></td>
      `
      const actions = tr.querySelector('.td-actions')
      const openBtn = document.createElement('button')
      openBtn.className = 'btn small'
      openBtn.textContent = 'Open'
      openBtn.addEventListener('click', () => openDetails(item))
      actions.appendChild(openBtn)
    }

    tbody.appendChild(tr)
  })

  table.appendChild(tbody)
  container.appendChild(table)
}

function onAnswerProblem(problem)
{
  const title = window.prompt('Provide a short title for your answer:')
  if (!title)
  {
    return
  }
  const id = `s${Date.now()}`
  placeholderGraph.nodes.push({ id, type: 'solution', label: title, reuseScore: 0.0, upvotes: 0 })
  placeholderGraph.links.push({ source: problem.id, target: id, type: 'solves', strength: 1.0 })

  state.seed += 1
  updateScoresUI()
  toast('Answer submitted. +1 Seed')

  if (window.__Graph)
  {
    window.__Graph.graphData(placeholderGraph)
  }
  renderList()
}

function initRetrieveBar()
{
  const retrieveBtn = document.getElementById('retrieveButton')
  if (retrieveBtn)
  {
    retrieveBtn.addEventListener('click', () =>
    {
      const q = document.getElementById('retrieveInput').value.trim()
      if (!q)
      {
        return
      }
      state.activeTab = 'problems'
      document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === 'problems'))
      renderList(q.toLowerCase())
    })
  }
}

function initDetailsPanel()
{
  document.getElementById('closeDetails').addEventListener('click', closeDetails)
}

async function init()
{
  initGraph()
  initHeader()
  initTabs()
  initList()
  initRetrieveBar()
  initDetailsPanel()
  
  // Load real graph data from KuzuDB
  await loadGraphData()
}

window.addEventListener('DOMContentLoaded', init) 