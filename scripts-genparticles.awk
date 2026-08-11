# genparticles.awk — bakes a particles.js-style network into pure SVG.
#
# particles.js recomputes, every frame, which nodes are close enough to link.
# SVG has no scripting, so that has to be precomputed: simulate the motion,
# sample it at K keyframes, and emit each edge as a <line> whose four endpoint
# attributes and stroke-opacity are all animated through those samples. An edge
# beyond the link distance is simply given opacity 0 at that keyframe, so links
# fade in and out exactly as they do on the site.
#
# Every node travels a closed ellipse whose angular period is an integer
# multiple of the loop, so position at t=1 equals position at t=0 and the whole
# animation repeats with no seam. keyTimes are omitted deliberately: SMIL
# distributes values evenly by default, which is what we want and saves ~30%
# of the file size.
#
#   awk -v SEED=3 -v N=28 -v K=18 -v T=24 -v L=155 -f genparticles.awk /dev/null

function abs(v) { return v < 0 ? -v : v }

BEGIN {
  PI = 3.14159265358979
  srand(SEED + 0)
  if (N == 0) N = 28
  if (K == 0) K = 18
  if (T == 0) T = 24
  if (L == 0) L = 155
  if (COL == "")  COL  = "#CECBF6"
  if (LCOL == "") LCOL = "#8E7BFF"
  if (LOPA == 0)  LOPA = 0.55
  if (NOPA == 0)  NOPA = 0.8

  X0 = -50; Y0 = -40; W = 1380; H = 420

  # --- lay out the nodes -------------------------------------------------
  # Reject points that land too near an existing one; even spacing is what
  # makes it read as a lattice rather than a smear.
  n = 0
  for (tries = 0; tries < N * 500 && n < N; tries++) {
    px = X0 + rand() * W
    py = Y0 + rand() * H
    ok = 1
    for (j = 1; j <= n; j++) {
      dx = px - BX[j]; dy = py - BY[j]
      if (dx*dx + dy*dy < 118*118) { ok = 0; break }
    }
    if (!ok) continue
    n++
    BX[n] = px; BY[n] = py
    RX[n] = 16 + rand() * 44          # ellipse radii
    RY[n] = 12 + rand() * 32
    P1[n] = rand() * 2 * PI           # phases
    P2[n] = rand() * 2 * PI
    M[n]  = (rand() < 0.62) ? 1 : 2   # integer periods => seamless loop
    NR_[n] = 1.2 + rand() * 1.5       # node radius
  }

  # --- sample the trajectories -------------------------------------------
  for (k = 0; k <= K; k++) {
    t = k / K
    for (i = 1; i <= n; i++) {
      a = 2 * PI * M[i] * t
      PX[i, k] = BX[i] + RX[i] * cos(a + P1[i])
      PY[i, k] = BY[i] + RY[i] * sin(a + P2[i])
    }
  }

  # --- edges --------------------------------------------------------------
  printf "        <g stroke=\"%s\" stroke-width=\"0.75\" fill=\"none\">\n", LCOL
  for (i = 1; i <= n; i++) {
    for (j = i + 1; j <= n; j++) {
      live = 0
      for (k = 0; k <= K; k++) {
        dx = PX[i,k] - PX[j,k]; dy = PY[i,k] - PY[j,k]
        D[k] = sqrt(dx*dx + dy*dy)
        if (D[k] < L) live = 1
      }
      if (!live) continue
      if (DEG[i] >= 5 || DEG[j] >= 5) continue
      DEG[i]++; DEG[j]++
      edges++

      x1s = ""; y1s = ""; x2s = ""; y2s = ""; os = ""
      for (k = 0; k <= K; k++) {
        sep = (k ? ";" : "")
        x1s = x1s sep sprintf("%.0f", PX[i,k])
        y1s = y1s sep sprintf("%.0f", PY[i,k])
        x2s = x2s sep sprintf("%.0f", PX[j,k])
        y2s = y2s sep sprintf("%.0f", PY[j,k])
        o = (D[k] >= L) ? 0 : (1 - D[k]/L) * LOPA
        os = os sep sprintf("%.2f", o)
      }
      printf "          <line stroke-opacity=\"0\">"
      printf "<animate attributeName=\"x1\" values=\"%s\" dur=\"%ds\" repeatCount=\"indefinite\"/>", x1s, T
      printf "<animate attributeName=\"y1\" values=\"%s\" dur=\"%ds\" repeatCount=\"indefinite\"/>", y1s, T
      printf "<animate attributeName=\"x2\" values=\"%s\" dur=\"%ds\" repeatCount=\"indefinite\"/>", x2s, T
      printf "<animate attributeName=\"y2\" values=\"%s\" dur=\"%ds\" repeatCount=\"indefinite\"/>", y2s, T
      printf "<animate attributeName=\"stroke-opacity\" values=\"%s\" dur=\"%ds\" repeatCount=\"indefinite\"/>", os, T
      printf "</line>\n"
    }
  }
  printf "        </g>\n"

  # --- nodes --------------------------------------------------------------
  printf "        <g fill=\"%s\">\n", COL
  for (i = 1; i <= n; i++) {
    xs = ""; ys = ""
    for (k = 0; k <= K; k++) {
      sep = (k ? ";" : "")
      xs = xs sep sprintf("%.0f", PX[i,k])
      ys = ys sep sprintf("%.0f", PY[i,k])
    }
    printf "          <circle r=\"%.1f\" opacity=\"%.2f\">", NR_[i], NOPA
    printf "<animate attributeName=\"cx\" values=\"%s\" dur=\"%ds\" repeatCount=\"indefinite\"/>", xs, T
    printf "<animate attributeName=\"cy\" values=\"%s\" dur=\"%ds\" repeatCount=\"indefinite\"/>", ys, T
    if (i % 6 == 0)
      printf "<animate attributeName=\"opacity\" values=\"%.2f;%.2f;%.2f\" dur=\"%ds\" repeatCount=\"indefinite\"/>", \
             NOPA, NOPA*0.2, NOPA, 5 + (i % 4)
    printf "</circle>\n"
  }
  printf "        </g>\n"

  printf "<!-- nodes=%d edges=%d keyframes=%d loop=%ds -->\n", n, edges, K, T
}
