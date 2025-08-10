/* global ForceGraph3D */

// Basic placeholder data schema kept flexible for later changes
// Node types: problem | solution | model
// Link types: similarity | solves | produced_by | related

const placeholderGraph = {
  nodes: [
    { id: 'p1', type: 'problem', label: 'Minify JS without breaking React keys', domain: 'Code', tags: ['react', 'build'], karma: 12 },
    { id: 'p2', type: 'problem', label: 'Extract entities from legal contracts', domain: 'Research', tags: ['nlp', 'contracts'], karma: 9 },
    { id: 's1', type: 'solution', label: 'AST-aware minifier config', reuseScore: 0.86, cost: '$', upvotes: 17 },
    { id: 's2', type: 'solution', label: 'Few-shot contract NER', reuseScore: 0.78, cost: '$$', upvotes: 33 },
    { id: 'm1', type: 'model', label: 'Llama 3 70B', size: 'Large', provider: 'Meta' },
    { id: 'm2', type: 'model', label: 'GPT-4o', size: 'XL', provider: 'OpenAI' },
    { id: 'm3', type: 'model', label: 'Phi-3 Mini', size: 'Small', provider: 'Microsoft' },
  ],
  links: [
    { source: 'p1', target: 's1', type: 'solves', strength: 1.0 },
    { source: 'p2', target: 's2', type: 'solves', strength: 1.0 },
    { source: 's1', target: 'm3', type: 'produced_by', strength: 1.0 },
    { source: 's2', target: 'm2', type: 'produced_by', strength: 1.0 },
    { source: 'p1', target: 'p2', type: 'similarity', strength: 0.28 },
    { source: 's1', target: 's2', type: 'related', strength: 0.24 },
  ]
}

const state = {
  showLabels: false,
  pinnedNodes: new Set(),
  activeTab: 'problems',
  sortBy: 'relevance',
  karma: Number(window.localStorage.getItem('karmaScore') || 42)
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
  return '#60a5fa'
}

function linkColor(link)
{
  if (link.type === 'similarity')
  {
    return 'rgba(122,162,255,0.8)'
  }
  if (link.type === 'solves')
  {
    return 'rgba(52,211,153,0.9)'
  }
  if (link.type === 'produced_by')
  {
    return 'rgba(96,165,250,0.9)'
  }
  return 'rgba(234,179,8,0.9)'
}

function initGraph()
{
  const elem = document.getElementById('graph')
  const Graph = ForceGraph3D()(elem)
    .graphData(placeholderGraph)
    .nodeAutoColorBy('type')
    .nodeVal(node => node.type === 'model' ? 9 : node.type === 'solution' ? 7 : 6)
    .nodeColor(node => nodeColor(node))
    .nodeOpacity(0.95)
    .nodeThreeObjectExtend(true)
    .nodeLabel(node => state.showLabels ? node.label : '')
    .linkColor(link => linkColor(link))
    .linkOpacity(0.35)
    .linkCurvature(link => 0.35 + Math.random() * 0.3) // variable curvature like example
    .linkDirectionalParticles(2)
    .linkDirectionalParticleWidth(link => Math.max(1, (link.strength || 0.2) * 4))
    .linkDirectionalParticleSpeed(0.005)
    .backgroundColor('#0b0d12')
    .onNodeClick(node => onNodeClick(node, Graph))
    .onBackgroundClick(() => closeDetails())

  // initial camera position for a cinematic reveal
  Graph.cameraPosition({ x: 0, y: 0, z: 180 }, { x: 0, y: 0, z: 0 }, 1500)

  // expose for debug
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

  title.textContent = node.label || 'Details'

  const rows = []
  Object.keys(node).forEach(key =>
  {
    if (['x', 'y', 'z', 'vx', 'vy', 'vz', 'fx', 'fy', 'fz'].includes(key))
    {
      return
    }
    rows.push(`<div class="k">${key}</div><div class="v">${formatValue(node[key])}</div>`)
  })

  content.innerHTML = `<div class="kv">${rows.join('')}</div>`
  panel.classList.remove('hidden')
}

function closeDetails()
{
  document.getElementById('detailsPanel').classList.add('hidden')
}

function formatValue(value)
{
  if (Array.isArray(value))
  {
    return value.join(', ')
  }
  if (typeof value === 'object' && value !== null)
  {
    return JSON.stringify(value)
  }
  return String(value)
}

function initHeader()
{
  document.getElementById('karmaScore').textContent = state.karma
  document.getElementById('toggleLabels').addEventListener('click', () =>
  {
    state.showLabels = !state.showLabels
    if (window.__Graph)
    {
      window.__Graph.nodeLabel(node => state.showLabels ? node.label : '')
    }
  })
  document.getElementById('resetCamera').addEventListener('click', () =>
  {
    if (window.__Graph)
    {
      window.__Graph.cameraPosition({ x: 0, y: 0, z: 180 }, { x: 0, y: 0, z: 0 }, 1200)
    }
  })
  document.getElementById('pinSelected').addEventListener('click', () =>
  {
    // Visual placeholder: toggles pin on the last clicked node by freezing position
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

  document.getElementById('searchButton').addEventListener('click', () =>
  {
    const q = document.getElementById('globalSearch').value.trim().toLowerCase()
    renderList(q)
  })
  document.getElementById('globalSearch').addEventListener('keydown', e =>
  {
    if (e.key === 'Enter')
    {
      e.preventDefault()
      const q = e.currentTarget.value.trim().toLowerCase()
      renderList(q)
    }
  })
  document.getElementById('uploadCta').addEventListener('click', () =>
  {
    // Placeholder: simulate successful upload (+1 karma)
    state.karma += 1
    document.getElementById('karmaScore').textContent = state.karma
    window.localStorage.setItem('karmaScore', String(state.karma))
    toast('Upload received. Karma +1')
  })
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
  document.getElementById('sortSelect').addEventListener('change', e =>
  {
    state.sortBy = e.target.value
    renderList()
  })
  renderList()
}

function getItemsForTab()
{
  if (state.activeTab === 'problems')
  {
    return placeholderGraph.nodes.filter(n => n.type === 'problem')
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

function renderList(query = '')
{
  const container = document.getElementById('listContainer')
  container.innerHTML = ''
  const items = sortItems(getItemsForTab())
  const filtered = items.filter(item =>
  {
    if (!query)
    {
      return true
    }
    const hay = `${item.label || ''} ${(item.tags || []).join(' ')} ${(item.domain || '')}`.toLowerCase()
    return hay.includes(query)
  })

  const fragments = document.createDocumentFragment()
  filtered.forEach(item =>
  {
    const card = document.createElement('div')
    card.className = 'card'

    const header = document.createElement('div')
    header.className = 'card-header'
    header.innerHTML = `<div class="card-title">${item.label}</div>` +
      `<div class="card-meta">` +
      (item.type ? `<span class="badge">${item.type}</span>` : '') +
      (item.domain ? `<span class="badge">${item.domain}</span>` : '') +
      `</div>`

    const metrics = document.createElement('div')
    metrics.className = 'metrics'
    metrics.innerHTML = `
      ${item.reuseScore != null ? `<span>Reuse: ${(item.reuseScore * 100).toFixed(0)}%</span>` : ''}
      ${item.karma != null ? `<span>Karma: ${item.karma}</span>` : ''}
      ${item.upvotes != null ? `<span>Upvotes: ${item.upvotes}</span>` : ''}
      ${item.cost ? `<span>Cost: ${item.cost}</span>` : ''}
      ${item.size ? `<span>Size: ${item.size}</span>` : ''}
    `

    const actions = document.createElement('div')
    actions.className = 'actions'
    const openBtn = document.createElement('button')
    openBtn.className = 'btn small'
    openBtn.textContent = 'Open details'
    openBtn.addEventListener('click', () => openDetails(item))

    const upvoteBtn = document.createElement('button')
    upvoteBtn.className = 'btn small'
    upvoteBtn.textContent = 'Upvote'
    upvoteBtn.addEventListener('click', () =>
    {
      state.karma += 1
      document.getElementById('karmaScore').textContent = state.karma
      window.localStorage.setItem('karmaScore', String(state.karma))
      toast('Upvoted. Karma +1')
    })

    actions.appendChild(openBtn)
    actions.appendChild(upvoteBtn)

    card.appendChild(header)
    card.appendChild(metrics)
    card.appendChild(actions)

    fragments.appendChild(card)
  })

  container.appendChild(fragments)
}

function initRetrieveBar()
{
  document.getElementById('retrieveButton').addEventListener('click', () =>
  {
    const q = document.getElementById('retrieveInput').value.trim()
    if (!q)
    {
      return
    }
    // For now, just filter list and focus Problems tab
    state.activeTab = 'problems'
    document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === 'problems'))
    renderList(q.toLowerCase())
  })
}

function initDetailsPanel()
{
  document.getElementById('closeDetails').addEventListener('click', closeDetails)
}

function init()
{
  initGraph()
  initHeader()
  initTabs()
  initList()
  initRetrieveBar()
  initDetailsPanel()
}

window.addEventListener('DOMContentLoaded', init) 