(function()
{
  var seeders = Number(window.localStorage.getItem('seedScore') || 8)
  var leechers = Number(window.localStorage.getItem('leechScore') || 2)

  var seedersEl = document.getElementById('seedersCount')
  var leechersEl = document.getElementById('leechersCount')
  var ratioText = document.getElementById('ratioText')
  var seedersBar = document.getElementById('seedersBar')
  var leechersBar = document.getElementById('leechersBar')
  var donut = document.getElementById('ratioDonut')
  var spark = document.getElementById('sparkline')
  var ctx = spark.getContext('2d')

  var rateHistory = []
  for (var i = 0; i < 60; i++)
  {
    rateHistory.push({ s: seeders, l: leechers })
  }

  function clamp(n, min, max)
  {
    return Math.max(min, Math.min(max, n))
  }

  function updateNumbers()
  {
    // Simulate more seeding than leeching
    var sDelta = Math.random() < 0.8 ? 1 : 0
    var lDelta = Math.random() < 0.4 ? 1 : 0
    seeders += sDelta
    leechers += lDelta

    window.localStorage.setItem('seedScore', String(seeders))
    window.localStorage.setItem('leechScore', String(leechers))

    var total = seeders + leechers
    var sPct = total > 0 ? Math.round((seeders / total) * 100) : 0
    var lPct = 100 - sPct

    if (seedersEl) { seedersEl.textContent = seeders.toLocaleString() }
    if (leechersEl) { leechersEl.textContent = leechers.toLocaleString() }
    if (ratioText) { ratioText.textContent = `${seeders}:${leechers}` }
    if (seedersBar) { seedersBar.style.width = clamp(sPct, 0, 100) + '%' }
    if (leechersBar) { leechersBar.style.width = clamp(lPct, 0, 100) + '%' }

    drawDonut(sPct)

    rateHistory.push({ s: seeders, l: leechers })
    if (rateHistory.length > 120) { rateHistory.shift() }
    drawSparkline()
  }

  function drawDonut(sPct)
  {
    if (!donut) { return }
    donut.innerHTML = ''
    var size = donut.clientHeight
    var svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg')
    svg.setAttribute('width', size)
    svg.setAttribute('height', size)
    svg.setAttribute('viewBox', `0 0 ${size} ${size}`)
    var radius = size / 2 - 8
    var center = size / 2

    // background circle
    var bg = document.createElementNS('http://www.w3.org/2000/svg', 'circle')
    bg.setAttribute('cx', center)
    bg.setAttribute('cy', center)
    bg.setAttribute('r', radius)
    bg.setAttribute('stroke', 'rgba(255,255,255,0.15)')
    bg.setAttribute('stroke-width', '10')
    bg.setAttribute('fill', 'none')
    svg.appendChild(bg)

    var circ = 2 * Math.PI * radius
    var sLen = (sPct / 100) * circ

    var arc = document.createElementNS('http://www.w3.org/2000/svg', 'circle')
    arc.setAttribute('cx', center)
    arc.setAttribute('cy', center)
    arc.setAttribute('r', radius)
    arc.setAttribute('stroke', 'url(#g1)')
    arc.setAttribute('stroke-width', '10')
    arc.setAttribute('fill', 'none')
    arc.setAttribute('stroke-dasharray', `${sLen} ${circ - sLen}`)
    arc.setAttribute('transform', `rotate(-90 ${center} ${center})`)

    // gradient def (reuse id g1 if exists)
    var defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs')
    var lg = document.createElementNS('http://www.w3.org/2000/svg', 'linearGradient')
    lg.setAttribute('id', 'g1')
    lg.setAttribute('x1', '0')
    lg.setAttribute('y1', '0')
    lg.setAttribute('x2', '1')
    lg.setAttribute('y2', '1')
    var stop1 = document.createElementNS('http://www.w3.org/2000/svg', 'stop')
    stop1.setAttribute('offset', '0%')
    stop1.setAttribute('stop-color', '#7aa2ff')
    var stop2 = document.createElementNS('http://www.w3.org/2000/svg', 'stop')
    stop2.setAttribute('offset', '100%')
    stop2.setAttribute('stop-color', '#22d3ee')
    lg.appendChild(stop1)
    lg.appendChild(stop2)
    defs.appendChild(lg)

    svg.appendChild(defs)
    svg.appendChild(arc)
    donut.appendChild(svg)
  }

  function drawSparkline()
  {
    if (!ctx) { return }
    var w = spark.width
    var h = spark.height
    ctx.clearRect(0, 0, w, h)

    var maxDelta = 20
    // compute recent deltas for seeders minus leechers to express health
    var points = []
    for (var i = Math.max(0, rateHistory.length - 60); i < rateHistory.length; i++)
    {
      var prev = rateHistory[i - 1] || rateHistory[i]
      var cur = rateHistory[i]
      var delta = (cur.s - prev.s) - (cur.l - prev.l)
      points.push(delta)
    }
    var maxAbs = Math.max(1, Math.min(maxDelta, Math.max.apply(null, points.map(function(x){ return Math.abs(x) }))))

    ctx.strokeStyle = '#7aa2ff'
    ctx.beginPath()
    var stepX = w / Math.max(1, points.length - 1)
    for (var j = 0; j < points.length; j++)
    {
      var x = j * stepX
      var y = h / 2 - (points[j] / maxAbs) * (h / 2 - 10)
      if (j === 0) { ctx.moveTo(x, y) } else { ctx.lineTo(x, y) }
    }
    ctx.stroke()
  }

  function tick()
  {
    updateNumbers()
  }

  updateNumbers()
  setInterval(tick, 1200)
})(); 