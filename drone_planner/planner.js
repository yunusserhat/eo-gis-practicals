// Copyright (C) 2026 Yunus Serhat Bıçakçı
// SPDX-License-Identifier: GPL-3.0-or-later
// Drone survey planner. All geometry is worked out in the British National Grid (metres), then drawn on a web map.
(function () {
  "use strict";

  const T = window.TERRAIN;
  const CELL = T.cell;
  const UK_LIMIT = 120;      // metres above the ground in the UK open category
  const LOW_CLEARANCE = 30;  // metres, below this trees and buildings may be in the way
  const TURN_SECONDS = 6;
  const CLIMB_RATE = 3;      // metres per second, used for climb and descent time
  const BATTERY_USE = 0.8;

  // Typical cameras. All values can be edited on the page.
  const CAMERAS = [
    { name: "1-inch 20 MP mapping camera", sensorW: 13.2, sensorH: 8.8, focal: 8.8, imgW: 5472, imgH: 3648, minInterval: 2 },
    { name: "4/3-inch 20 MP camera", sensorW: 17.3, sensorH: 13.0, focal: 12.3, imgW: 5280, imgH: 3956, minInterval: 0.7 },
    { name: "Small sub-250 g drone, 12 MP mode", sensorW: 9.6, sensorH: 7.2, focal: 6.7, imgW: 4032, imgH: 3024, minInterval: 2 },
  ];

  const sites = T.sites.map(decodeSite);

  function decodeSite(s) {
    const bin = atob(s.elevation_dm);
    const view = new DataView(new ArrayBuffer(bin.length));
    for (let i = 0; i < bin.length; i++) view.setUint8(i, bin.charCodeAt(i));
    const z = new Float32Array(bin.length / 2);
    for (let i = 0; i < z.length; i++) z[i] = view.getInt16(2 * i, true) / 10;
    return Object.assign({}, s, { z: z });
  }

  // ---------- coordinates ----------
  function toLatLng(site, e, n) {
    const de = e - site.bounds[0], dn = n - site.bounds[1];
    const lo = site.lonlat.lon, la = site.lonlat.lat;
    return [la[0] + la[1] * de + la[2] * dn, lo[0] + lo[1] * de + lo[2] * dn];
  }

  function toEN(site, lat, lon) {
    const lo = site.lonlat.lon, la = site.lonlat.lat;
    const a = lo[1], b = lo[2], c = la[1], d = la[2];
    const det = a * d - b * c;
    const x = lon - lo[0], y = lat - la[0];
    return [site.bounds[0] + (d * x - b * y) / det, site.bounds[1] + (-c * x + a * y) / det];
  }

  function elevationAt(site, e, n) {
    const fx = (e - site.bounds[0]) / CELL - 0.5;
    const fy = (site.bounds[3] - n) / CELL - 0.5;
    const x0 = Math.max(0, Math.min(site.nx - 1, Math.floor(fx)));
    const y0 = Math.max(0, Math.min(site.ny - 1, Math.floor(fy)));
    const x1 = Math.min(site.nx - 1, x0 + 1), y1 = Math.min(site.ny - 1, y0 + 1);
    const tx = Math.min(1, Math.max(0, fx - x0)), ty = Math.min(1, Math.max(0, fy - y0));
    const z = site.z, nx = site.nx;
    const top = z[y0 * nx + x0] * (1 - tx) + z[y0 * nx + x1] * tx;
    const bottom = z[y1 * nx + x0] * (1 - tx) + z[y1 * nx + x1] * tx;
    return top * (1 - ty) + bottom * ty;
  }

  // ---------- the plan ----------
  function makePlan(p, site) {
    const H = p.height;
    const gsdNominal = H * p.sensorW / (p.focal * p.imgW);           // metres per pixel
    const across = H * p.sensorW / p.focal;                           // footprint across the flight line (m)
    const along = H * p.sensorH / p.focal;                            // footprint along the flight line (m)
    const lineSpacing = across * (1 - p.side / 100);
    const photoSpacing = along * (1 - p.front / 100);
    const interval = photoSpacing / p.speed;
    const blur = p.speed * p.shutter / gsdNominal;

    // Flight lines across the rectangular site, in a serpentine order.
    const th = p.heading * Math.PI / 180;
    const d = [Math.sin(th), Math.cos(th)];   // along the lines (east, north)
    const q = [Math.cos(th), -Math.sin(th)];  // across the lines
    const [e0, n0, e1, n1] = site.bounds;
    const ce = (e0 + e1) / 2, cn = (n0 + n1) / 2;
    const corners = [[e0, n0], [e1, n0], [e1, n1], [e0, n1]];
    const qs = corners.map(c => (c[0] - ce) * q[0] + (c[1] - cn) * q[1]);
    const qmin = Math.min.apply(null, qs), qmax = Math.max.apply(null, qs);
    const nPositions = Math.ceil((qmax - qmin) / lineSpacing) + 1;
    const segments = [];
    for (let k = 0; k < nPositions; k++) {
      const off = Math.min(qmin + k * lineSpacing, qmax);
      const px = ce + off * q[0], py = cn + off * q[1];
      let tmin = -Infinity, tmax = Infinity;
      [[px, d[0], e0, e1], [py, d[1], n0, n1]].forEach(function (axis) {
        const start = axis[0], dir = axis[1], lo = axis[2], hi = axis[3];
        if (Math.abs(dir) < 1e-9) {
          if (start < lo || start > hi) { tmin = Infinity; tmax = -Infinity; }
        } else {
          const ta = (lo - start) / dir, tb = (hi - start) / dir;
          tmin = Math.max(tmin, Math.min(ta, tb));
          tmax = Math.min(tmax, Math.max(ta, tb));
        }
      });
      if (tmax - tmin > 1) {
        let a = [px + tmin * d[0], py + tmin * d[1]], b = [px + tmax * d[0], py + tmax * d[1]];
        if (segments.length % 2 === 1) { const t = a; a = b; b = t; }
        segments.push([a, b]);
      }
    }

    const takeoff = [site.takeoff.e, site.takeoff.n];
    const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
    let lineLength = 0, transit = 0, photos = 0;
    segments.forEach(function (s, i) {
      const len = dist(s[0], s[1]);
      lineLength += len;
      photos += Math.floor(len / photoSpacing) + 1;
      if (i > 0) transit += dist(segments[i - 1][1], s[0]);
    });
    if (segments.length) transit += dist(takeoff, segments[0][0]) + dist(segments[segments.length - 1][1], takeoff);
    const turns = Math.max(0, segments.length - 1);
    const seconds = (lineLength + transit) / p.speed + turns * TURN_SECONDS + 2 * H / CLIMB_RATE;
    const batteries = Math.ceil(seconds / (p.endurance * 60 * BATTERY_USE));

    // Height above the ground over every 5 m cell of the site.
    const zt = site.takeoff.elevation;
    const agl = new Float32Array(site.z.length);
    let aglMin = Infinity, aglMax = -Infinity, aglMinAbove = Infinity, aglMinSafe = Infinity, nBelow = 0, nLow = 0, nHigh = 0;
    for (let i = 0; i < site.z.length; i++) {
      const a = p.mode === "terrain" ? H : zt + H - site.z[i];
      agl[i] = a;
      if (a < aglMin) aglMin = a;
      if (a > aglMax) aglMax = a;
      if (a > 0 && a < aglMinAbove) aglMinAbove = a;
      if (a >= LOW_CLEARANCE && a < aglMinSafe) aglMinSafe = a;
      if (a <= 0) nBelow++; else if (a < LOW_CLEARANCE) nLow++;
      if (a > UK_LIMIT) nHigh++;
    }
    const gsdOf = a => a * p.sensorW / (p.focal * p.imgW);
    // Over the highest ground the footprint shrinks, so neighbouring lines overlap less. Undefined if the drone is
    // below the ground somewhere.
    const sideOverHighGround = aglMin > 0 ? 1 - lineSpacing / (aglMin * p.sensorW / p.focal) : null;

    // Profile along the flight line that crosses the highest ground.
    let best = null;
    segments.forEach(function (s) {
      const len = dist(s[0], s[1]);
      const steps = Math.max(2, Math.ceil(len / 5));
      const pts = [];
      let top = -Infinity;
      for (let j = 0; j <= steps; j++) {
        const t = j / steps;
        const e = s[0][0] + t * (s[1][0] - s[0][0]), n = s[0][1] + t * (s[1][1] - s[0][1]);
        const g = elevationAt(site, e, n);
        top = Math.max(top, g);
        pts.push({ d: t * len, ground: g, drone: p.mode === "terrain" ? g + H : zt + H });
      }
      if (!best || top > best.top) best = { top: top, pts: pts, seg: s };
    });

    return {
      gsdNominal, across, along, lineSpacing, photoSpacing, interval, blur, segments, photos, turns,
      distance: lineLength + transit, seconds, batteries, agl, aglMin, aglMax,
      // Pixel sizes are reported where the drone can fly safely, at least LOW_CLEARANCE above the ground.
      gsdMin: gsdOf(isFinite(aglMinSafe) ? aglMinSafe : (isFinite(aglMinAbove) ? aglMinAbove : 0)),
      gsdMax: gsdOf(Math.max(aglMax, 0)),
      shareBelow: nBelow / agl.length, shareLow: nLow / agl.length, shareHigh: nHigh / agl.length,
      sideOverHighGround, profile: best,
    };
  }

  // ---------- map ----------
  const map = L.map("map", { zoomControl: true });
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19, attribution: "© OpenStreetMap contributors",
  }).addTo(map);
  const layers = L.layerGroup().addTo(map);
  let overlayLayer = null;
  let currentSiteId = null;

  const RAMP = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
  function rampColour(t) {
    t = Math.min(1, Math.max(0, t)) * (RAMP.length - 1);
    const i = Math.min(RAMP.length - 2, Math.floor(t)), f = t - i;
    return RAMP[i].map((c, k) => Math.round(c + f * (RAMP[i + 1][k] - c)));
  }

  function hillshade(site) {
    const out = new Float32Array(site.z.length), nx = site.nx, ny = site.ny, z = site.z;
    // Light from the north-west, 45 degrees above the horizon, as a unit vector (east, north, up).
    const az = 315 * Math.PI / 180, alt = 45 * Math.PI / 180;
    const lx = Math.sin(az) * Math.cos(alt), ly = Math.cos(az) * Math.cos(alt), lz = Math.sin(alt);
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        const xl = Math.max(0, x - 1), xr = Math.min(nx - 1, x + 1), yn = Math.max(0, y - 1), ys = Math.min(ny - 1, y + 1);
        const dzde = (z[y * nx + xr] - z[y * nx + xl]) / ((xr - xl) * CELL);  // slope towards the east
        const dzdn = (z[yn * nx + x] - z[ys * nx + x]) / ((ys - yn) * CELL);  // slope towards the north
        const v = (-dzde * lx - dzdn * ly + lz) / Math.hypot(dzde, dzdn, 1);
        out[y * nx + x] = Math.max(0, v);
      }
    }
    return out;
  }

  // Draw a grid that lives in the National Grid onto a north-up image for the web map.
  function warpToImage(site, colourAt) {
    const [e0, n0, e1, n1] = site.bounds;
    const cs = [[e0, n0], [e1, n0], [e1, n1], [e0, n1]].map(c => toLatLng(site, c[0], c[1]));
    const south = Math.min.apply(null, cs.map(c => c[0])), north = Math.max.apply(null, cs.map(c => c[0]));
    const west = Math.min.apply(null, cs.map(c => c[1])), east = Math.max.apply(null, cs.map(c => c[1]));
    const w = site.nx * 2, h = site.ny * 2;
    const canvas = document.createElement("canvas");
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(w, h);
    for (let i = 0; i < h; i++) {
      const lat = north - (i + 0.5) / h * (north - south);
      for (let j = 0; j < w; j++) {
        const lon = west + (j + 0.5) / w * (east - west);
        const en = toEN(site, lat, lon);
        const col = Math.floor((en[0] - e0) / CELL), row = Math.floor((n1 - en[1]) / CELL);
        if (col < 0 || row < 0 || col >= site.nx || row >= site.ny) continue;
        const rgba = colourAt(row * site.nx + col);
        const k = 4 * (i * w + j);
        img.data[k] = rgba[0]; img.data[k + 1] = rgba[1]; img.data[k + 2] = rgba[2]; img.data[k + 3] = rgba[3];
      }
    }
    ctx.putImageData(img, 0, 0);
    return { url: canvas.toDataURL(), bounds: [[south, west], [north, east]] };
  }

  function drawMap(p, site, plan) {
    layers.clearLayers();
    if (overlayLayer) { map.removeLayer(overlayLayer); overlayLayer = null; }
    const legend = document.getElementById("legend");
    if (p.overlay === "gsd") {
      const g0 = plan.gsdMin * 100, g1 = Math.max(plan.gsdMax * 100, g0 + 0.2);
      const scale = p.sensorW / (p.focal * p.imgW) * 100;
      const img = warpToImage(site, function (i) {
        const a = plan.agl[i];
        if (a <= 0) return [20, 20, 20, 200];
        if (a < LOW_CLEARANCE) return [215, 48, 39, 190];
        if (a > UK_LIMIT) return [123, 50, 148, 170];
        const c = rampColour((a * scale - g0) / (g1 - g0));
        return [c[0], c[1], c[2], 150];
      });
      overlayLayer = L.imageOverlay(img.url, img.bounds, { opacity: 1 }).addTo(map);
      legend.innerHTML =
        "Pixel size <span>" + g0.toFixed(1) + " cm</span><span class='ramp' style='background:linear-gradient(90deg," +
        RAMP.map(c => "rgb(" + c.join(",") + ")").join(",") + ")'></span><span>" + g1.toFixed(1) + " cm</span>" +
        "<span><span class='swatch' style='background:#d73027'></span>under " + LOW_CLEARANCE + " m above ground</span>" +
        "<span><span class='swatch' style='background:#141414'></span>below the ground</span>" +
        "<span><span class='swatch' style='background:#7b3294'></span>over " + UK_LIMIT + " m above ground</span>";
    } else if (p.overlay === "hillshade") {
      const hs = site.hillshade || (site.hillshade = hillshade(site));
      const img = warpToImage(site, function (i) { const v = Math.round(255 * hs[i]); return [v, v, v, 170]; });
      overlayLayer = L.imageOverlay(img.url, img.bounds).addTo(map);
      legend.textContent = "Hillshade from the LiDAR terrain model, lit from the north-west.";
    } else {
      legend.textContent = "";
    }

    const [e0, n0, e1, n1] = site.bounds;
    L.polygon([[e0, n0], [e1, n0], [e1, n1], [e0, n1]].map(c => toLatLng(site, c[0], c[1])),
      { color: "#1f2933", weight: 2, fill: false, dashArray: "6 4" }).addTo(layers);
    const route = [];
    plan.segments.forEach(function (s, i) {
      if (i > 0) L.polyline([toLatLng(site, plan.segments[i - 1][1][0], plan.segments[i - 1][1][1]), toLatLng(site, s[0][0], s[0][1])],
        { color: "#ffffff", weight: 2, opacity: 0.9, dashArray: "3 4" }).addTo(layers);
      route.push(toLatLng(site, s[0][0], s[0][1]), toLatLng(site, s[1][0], s[1][1]));
      L.polyline([toLatLng(site, s[0][0], s[0][1]), toLatLng(site, s[1][0], s[1][1])],
        { color: "#ffffff", weight: 2.5, opacity: 0.95 }).addTo(layers);
    });
    if (plan.profile) {
      const s = plan.profile.seg;
      L.polyline([toLatLng(site, s[0][0], s[0][1]), toLatLng(site, s[1][0], s[1][1])],
        { color: "#ff8c00", weight: 4, opacity: 1 }).addTo(layers).bindTooltip("Line shown in the profile");
    }
    L.circleMarker(toLatLng(site, site.takeoff.e, site.takeoff.n),
      { radius: 7, color: "#ffffff", weight: 2, fillColor: "#b03a2e", fillOpacity: 1 })
      .addTo(layers).bindTooltip("Take-off, " + site.takeoff.elevation.toFixed(0) + " m above sea level");
    if (currentSiteId !== site.id) {
      currentSiteId = site.id;
      const box = L.latLngBounds([[e0, n0], [e1, n0], [e1, n1], [e0, n1]].map(c => toLatLng(site, c[0], c[1])));
      requestAnimationFrame(function () { map.invalidateSize(); map.fitBounds(box.pad(0.05)); });
    }
  }

  // ---------- profile ----------
  function drawProfile(p, plan) {
    const svg = document.getElementById("profile");
    const W = 720, H = 240, m = { l: 52, r: 14, t: 14, b: 34 };
    const pts = plan.profile ? plan.profile.pts : [];
    if (!pts.length) { svg.innerHTML = ""; return; }
    const len = pts[pts.length - 1].d;
    const ys = [];
    pts.forEach(q => { ys.push(q.ground, q.drone, Math.min(q.ground + UK_LIMIT, q.drone + 40)); });
    const ymin = Math.floor(Math.min.apply(null, ys) / 10) * 10 - 5, ymax = Math.ceil(Math.max.apply(null, ys) / 10) * 10 + 5;
    const X = d => m.l + d / len * (W - m.l - m.r);
    const Y = z => H - m.b - (z - ymin) / (ymax - ymin) * (H - m.t - m.b);
    const line = key => pts.map((q, i) => (i ? "L" : "M") + X(q.d).toFixed(1) + " " + Y(key(q)).toFixed(1)).join(" ");
    const ground = line(q => q.ground) + " L" + X(len) + " " + (H - m.b) + " L" + X(0) + " " + (H - m.b) + " Z";
    let ticks = "";
    const step = niceStep((ymax - ymin) / 5);
    for (let z = Math.ceil(ymin / step) * step; z <= ymax; z += step) {
      ticks += "<line x1='" + m.l + "' x2='" + (W - m.r) + "' y1='" + Y(z) + "' y2='" + Y(z) + "' stroke='#eef1f3'/>" +
        "<text x='" + (m.l - 6) + "' y='" + (Y(z) + 4) + "' text-anchor='end' font-size='11' fill='#5f6b76'>" + z + "</text>";
    }
    const xstep = niceStep(len / 6);
    for (let d = 0; d <= len; d += xstep) {
      ticks += "<text x='" + X(d) + "' y='" + (H - m.b + 16) + "' text-anchor='middle' font-size='11' fill='#5f6b76'>" + Math.round(d) + "</text>";
    }
    svg.innerHTML = ticks +
      "<path d='" + ground + "' fill='#c9b79c' stroke='#8a7350' stroke-width='1'/>" +
      "<path d='" + line(q => q.ground + UK_LIMIT) + "' fill='none' stroke='#7b3294' stroke-dasharray='5 4' stroke-width='1.5'/>" +
      "<path d='" + line(q => q.ground + LOW_CLEARANCE) + "' fill='none' stroke='#d73027' stroke-dasharray='2 3' stroke-width='1.2'/>" +
      "<path d='" + line(q => q.drone) + "' fill='none' stroke='#1b7f6a' stroke-width='2.5'/>" +
      "<text x='" + (W / 2) + "' y='" + (H - 4) + "' text-anchor='middle' font-size='11' fill='#5f6b76'>Distance along the line (m)</text>" +
      "<text x='14' y='" + (H / 2) + "' transform='rotate(-90 14 " + (H / 2) + ")' text-anchor='middle' font-size='11' fill='#5f6b76'>Height above sea level (m)</text>" +
      "<g font-size='11'><rect x='" + (W - 250) + "' y='" + (m.t + 2) + "' width='236' height='58' fill='white' opacity='0.85'/>" +
      "<line x1='" + (W - 242) + "' x2='" + (W - 222) + "' y1='" + (m.t + 14) + "' y2='" + (m.t + 14) + "' stroke='#1b7f6a' stroke-width='2.5'/><text x='" + (W - 216) + "' y='" + (m.t + 18) + "'>Drone</text>" +
      "<line x1='" + (W - 242) + "' x2='" + (W - 222) + "' y1='" + (m.t + 30) + "' y2='" + (m.t + 30) + "' stroke='#7b3294' stroke-dasharray='5 4' stroke-width='1.5'/><text x='" + (W - 216) + "' y='" + (m.t + 34) + "'>120 m above the ground (UK limit)</text>" +
      "<line x1='" + (W - 242) + "' x2='" + (W - 222) + "' y1='" + (m.t + 46) + "' y2='" + (m.t + 46) + "' stroke='#d73027' stroke-dasharray='2 3' stroke-width='1.2'/><text x='" + (W - 216) + "' y='" + (m.t + 50) + "'>30 m above the ground</text></g>";
  }

  function niceStep(raw) {
    const p = Math.pow(10, Math.floor(Math.log10(raw)));
    const f = raw / p;
    return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * p;
  }

  // ---------- results ----------
  function fmtTime(s) {
    const m = Math.floor(s / 60), r = Math.round(s % 60);
    return m + " min " + (r < 10 ? "0" : "") + r + " s";
  }

  function showResults(p, site, plan) {
    const rows = [
      ["Pixel size at the set height", (plan.gsdNominal * 100).toFixed(2) + " cm"],
      ["Pixel size where it is safe to fly", (plan.gsdMin * 100).toFixed(2) + " to " + (plan.gsdMax * 100).toFixed(2) + " cm"],
      ["Footprint of one photo", plan.across.toFixed(0) + " × " + plan.along.toFixed(0) + " m"],
      ["Distance between flight lines", plan.lineSpacing.toFixed(1) + " m"],
      ["Distance between photos", plan.photoSpacing.toFixed(1) + " m, every " + plan.interval.toFixed(1) + " s"],
      ["Flight lines", String(plan.segments.length)],
      ["Photos", plan.photos.toLocaleString("en-GB")],
      ["Flight distance", (plan.distance / 1000).toFixed(2) + " km"],
      ["Flight time", fmtTime(plan.seconds)],
      ["Batteries at 80% use", String(plan.batteries)],
      ["Motion blur", plan.blur.toFixed(2) + " px"],
      ["Height above the ground", Math.round(plan.aglMin) + " to " + Math.round(plan.aglMax) + " m"],
      ["Side overlap over the highest ground", plan.sideOverHighGround === null ? "not defined" :
        plan.sideOverHighGround < 0 ? "none, gaps" : Math.round(100 * plan.sideOverHighGround) + "%"],
    ];
    document.getElementById("results-table").innerHTML =
      rows.map(r => "<tr><td>" + r[0] + "</td><td>" + r[1] + "</td></tr>").join("");

    const notes = [];
    const pct = x => (100 * x < 1 ? "under 1" : Math.round(100 * x)) + "% of the site";
    if (plan.shareBelow > 0) notes.push(["warn", "The planned height is below the ground over " + pct(plan.shareBelow) + ". Follow the terrain or fly higher."]);
    if (plan.shareLow > 0) notes.push(["warn", "The drone would be less than " + LOW_CLEARANCE + " m above the ground over " + pct(plan.shareLow) + ", where trees and buildings may be in the way."]);
    if (plan.shareHigh > 0) notes.push(["warn", "The drone would be more than " + UK_LIMIT + " m above the ground over " + pct(plan.shareHigh) + ". The UK open category does not allow this."]);
    if (p.height > UK_LIMIT && p.mode === "terrain") notes.push(["warn", "The set height is above the " + UK_LIMIT + " m limit of the UK open category."]);
    if (plan.interval < p.minInterval) notes.push(["warn", "Photos are needed every " + plan.interval.toFixed(1) + " s, faster than the camera's " + p.minInterval + " s. Fly slower, fly higher or lower the front overlap."]);
    if (plan.blur > 1) notes.push(["warn", "Motion blur is above one pixel. Fly slower or use a faster shutter."]);
    else if (plan.blur > 0.5) notes.push(["warn", "Motion blur is above half a pixel, so fine detail may soften."]);
    if (plan.sideOverHighGround !== null && plan.sideOverHighGround < 0) notes.push(["warn", "Neighbouring flight lines no longer overlap over the highest ground, so the map will have gaps there."]);
    else if (plan.sideOverHighGround !== null && plan.sideOverHighGround < 0.3) notes.push(["warn", "Side overlap falls to " + Math.round(100 * plan.sideOverHighGround) + "% over the highest ground, so the map may have gaps there."]);
    if (plan.batteries > 1) notes.push(["warn", "The flight needs " + plan.batteries + " batteries, so plan " + (plan.batteries - 1) + " landing" + (plan.batteries > 2 ? "s" : "") + " to swap them."]);
    if (!notes.length) notes.push(["ok", "All checks passed for this plan."]);
    document.getElementById("warnings").innerHTML = notes.map(n => "<li class='" + n[0] + "'>" + n[1] + "</li>").join("");
  }

  // ---------- controls ----------
  const $ = id => document.getElementById(id);
  const siteSelect = $("site"), cameraSelect = $("camera");
  sites.forEach((s, i) => siteSelect.add(new Option(s.name, String(i))));
  siteSelect.value = String(Math.max(0, sites.findIndex(s => s.id === "cambuskenneth")));  // start on flat ground
  window.addEventListener("resize", function () { map.invalidateSize(); });
  CAMERAS.forEach((c, i) => cameraSelect.add(new Option(c.name, String(i))));

  function applyCamera() {
    const c = CAMERAS[+cameraSelect.value];
    ["sensorW", "sensorH", "focal", "imgW", "imgH", "minInterval"].forEach(k => { $(k).value = c[k]; });
  }

  function readParams() {
    return {
      height: +$("height").value,
      mode: document.querySelector("input[name='mode']:checked").value,
      front: +$("front").value, side: +$("side").value,
      speed: +$("speed").value, shutter: +$("shutter").value,
      heading: +$("heading").value, endurance: +$("endurance").value,
      sensorW: +$("sensorW").value, sensorH: +$("sensorH").value, focal: +$("focal").value,
      imgW: +$("imgW").value, imgH: +$("imgH").value, minInterval: +$("minInterval").value,
      overlay: $("overlay").value,
    };
  }

  function update() {
    const p = readParams();
    const site = sites[+siteSelect.value];
    $("site-note").textContent = site.note + " Take-off at " + site.takeoff.elevation.toFixed(0) + " m above sea level.";
    $("height-out").textContent = p.height + " m";
    $("front-out").textContent = p.front + "%";
    $("side-out").textContent = p.side + "%";
    $("speed-out").textContent = p.speed + " m/s";
    $("heading-out").textContent = p.heading + "° from grid north";
    $("endurance-out").textContent = p.endurance + " min";
    if (!(p.sensorW > 0 && p.sensorH > 0 && p.focal > 0 && p.imgW > 0)) return;
    const plan = makePlan(p, site);
    drawMap(p, site, plan);
    drawProfile(p, plan);
    showResults(p, site, plan);
  }

  cameraSelect.addEventListener("change", function () { applyCamera(); update(); });
  document.querySelectorAll("#controls input, #controls select").forEach(function (el) {
    if (el !== cameraSelect) el.addEventListener("input", update);
  });
  applyCamera();
  update();
  window.__planner = { sites: sites, makePlan: makePlan, readParams: readParams };
})();
