// ============================================================================
// Quantile Spine — white paper (accessible edition)
// Build:  typst compile docs/quantile-spine.typ
// ============================================================================

#import "@preview/cetz:0.4.2": canvas, draw

// ----------------------------------------------------------------------------
// Document setup
// ----------------------------------------------------------------------------
#set page(
  paper: "a4",
  margin: (x: 2.2cm, top: 2.4cm, bottom: 2.6cm),
  numbering: "1",
  footer: context [
    #set text(size: 8pt, fill: luma(120))
    #line(length: 100%, stroke: 0.4pt + luma(200))
    #v(-2pt)
    Quantile Spine — a compact distribution tracker for tensors
    #h(1fr)
    #counter(page).display("1")
  ],
)
#set text(size: 10.2pt, lang: "en")
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.")
#show heading: it => [#v(0.5em)#it#v(0.3em)]
#set math.equation(numbering: none)

// Palette
#let navy = rgb("#1e3a8a")
#let blue = rgb("#2563eb")
#let orange = rgb("#ea580c")
#let green = rgb("#059669")
#let purple = rgb("#7c3aed")
#let softgray = luma(120)

#show heading: set text(fill: navy)
#show link: set text(fill: blue)

// "Deeper dive" box for optional technical detail
#let deep(title, body) = block(
  width: 100%,
  fill: luma(248),
  stroke: (left: 2.5pt + navy),
  inset: 10pt,
  radius: 3pt,
  breakable: true,
)[
  #text(size: 8.5pt, weight: "bold", fill: navy)[#smallcaps[Deeper dive] — #title]
  #v(2pt)
  #set text(size: 9pt)
  #body
]

// Small citation marker, e.g. [1]
#let cite-mark(n) = text(size: 8pt, fill: blue)[#super[[#n]]]

// Plain-language callout
#let plain(body) = block(
  width: 100%,
  fill: rgb("#eff6ff"),
  inset: 10pt,
  radius: 4pt,
)[
  #set text(size: 9.5pt)
  #body
]

// ----------------------------------------------------------------------------
// Math helpers for figures (error function → normal CDF)
// ----------------------------------------------------------------------------
#let erf(x) = {
  let s = if x < 0 { -1 } else { 1 }
  let ax = calc.abs(x)
  let t = 1.0 / (1.0 + 0.3275911 * ax)
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
    - 0.284496736) * t + 0.254829592) * t * calc.exp(-ax * ax)
  s * y
}
#let normcdf(z) = 0.5 * (1.0 + erf(z / calc.sqrt(2.0)))

// ----------------------------------------------------------------------------
// Title
// ----------------------------------------------------------------------------
#align(center)[
  #v(0.5em)
  #text(size: 21pt, weight: "bold", fill: navy)[Quantile Spine]
  #v(0.2em)
  #text(size: 12.5pt, fill: luma(60))[
    A small, fast way to track millions of value distributions at once
  ]
  #v(0.4em)
  #text(size: 9.5pt, fill: softgray)[
    A design white paper for the *monatq* project · design, reasoning, and first head-to-head measurements
  ]
  #v(0.8em)
]

#line(length: 100%, stroke: 0.6pt + luma(200))

// ----------------------------------------------------------------------------
// One-minute summary
// ----------------------------------------------------------------------------
#plain[
  *The one-minute version.*
  When studying a neural network, we often want to know, for _every single
  position_ inside a tensor, what values tend to appear there: what is typical,
  what is rare, how extreme do things get? A tensor can have millions of
  positions, and each one sees a long stream of values — far too much to store.
  The current tool for this, the *t-digest*, works well but needs roughly
  *5 kilobytes of memory per position* and does complicated bookkeeping on
  every update.

  *Quantile Spine* is a redesign built on a simple observation: we never look
  at the bookkeeping — we only ever ask for the answers. So it stores the
  answers directly: a fixed set of *percentile marks* (the "spine"), an exact
  list of the record-lowest and record-highest values, and exact counters for
  special values such as zero. That is about *370 bytes per position — roughly
  13× smaller* — and every position is updated with the same simple arithmetic,
  which modern CPUs can do for many positions simultaneously. The sketch also
  *watches its own surprise*: every incoming batch is compared against what the
  spine expected, and disagreement makes the sketch adapt faster — so drifting
  or suddenly-changing streams are tracked instead of averaged away. A first
  implementation confirms the design (@sec-results): the predicted *13×*
  memory saving exactly, updates *1.2–1.9×* and queries *5× faster* than the
  t-digest, lower error — mean _and_ worst case — on every distribution
  tested, and spike- and drift-handling better by orders of magnitude — with
  extremes stored exactly.
]

// ============================================================================
= The problem: millions of tiny streams
// ============================================================================

A neural network processes data as *tensors* — large grids of numbers. Run the
network on one input and every position in the grid takes some value. Run it on
ten thousand inputs and every position has seen ten thousand values. Those
values form a little stream of history, one stream per position
(@fig-streams).

The *monatq* library exists to answer questions about those streams:

- _"What is the typical value at this position?"_ (the median)
- _"What range covers 98% of what this position sees?"_ (the 1st to 99th percentile)
- _"How extreme does it get?"_ (the record highs and lows)

These questions drive real decisions — for example, choosing how aggressively a
model can be compressed (quantization), or understanding which neurons behave
unusually (interpretability).

The catch is scale. Storing every value at every position would take
_number of samples × number of positions_ numbers — easily hundreds of
gigabytes. So each position must keep only a *tiny summary* of its stream, one
that can still answer percentile questions afterwards.

#figure(
  canvas(length: 1cm, {
    import draw: *
    let cell = 0.38
    let cols = 6
    let rows = 4
    // Three stacked tensor "sheets", drawn back to front
    for (si, off) in ((2, 0.56), (1, 0.28), (0, 0.0)).map(p => p) {
      let ox = off * 1.0
      let oy = off * 0.72
      let w = cols * cell
      let h = rows * cell
      rect((ox, oy), (ox + w, oy + h), fill: white, stroke: 0.6pt + luma(150))
      for i in range(1, cols) {
        line((ox + i * cell, oy), (ox + i * cell, oy + h), stroke: 0.3pt + luma(210))
      }
      for j in range(1, rows) {
        line((ox, oy + j * cell), (ox + w, oy + j * cell), stroke: 0.3pt + luma(210))
      }
      // highlight the same cell in every sheet
      rect(
        (ox + 3 * cell, oy + 1 * cell), (ox + 4 * cell, oy + 2 * cell),
        fill: blue.lighten(if si == 0 { 0% } else { 45% }), stroke: 0.5pt + navy,
      )
    }
    content((1.7, -0.55), text(size: 8pt, fill: softgray)[tensor samples over time])
    line((2.4, 2.35), (2.4, 2.9), stroke: 0.5pt + luma(150))
    content((2.4, 3.12), text(size: 8pt, fill: softgray)[same position, every sample])

    // arrow to the stream box
    line((3.6, 1.15), (4.55, 1.15), mark: (end: ">", fill: luma(90)), stroke: 0.9pt + luma(90))

    // stream of values box
    rect((4.7, 0.1), (8.6, 2.2), fill: luma(252), stroke: 0.6pt + luma(170), radius: 0.08)
    content((6.65, 1.95), text(size: 8pt, fill: softgray)[values seen at that position])
    let ys = (0.9, 1.3, 0.7, 1.1, 1.5, 0.85, 1.2, 0.6, 1.35, 1.0, 0.75, 1.45, 1.05, 0.95, 1.25)
    for (i, y) in ys.enumerate() {
      circle((4.95 + i * 0.24, y), radius: 0.045, fill: blue, stroke: none)
    }
    content((6.65, 0.34), text(size: 8pt, fill: softgray)[…one long stream per position])

    // arrow to summary
    line((8.75, 1.15), (9.7, 1.15), mark: (end: ">", fill: luma(90)), stroke: 0.9pt + luma(90))
    rect((9.85, 0.5), (12.75, 1.85), fill: rgb("#ecfdf5"), stroke: 0.7pt + green, radius: 0.08)
    content((11.3, 1.5), text(size: 8pt, fill: green.darken(20%))[*tiny summary*])
    content((11.3, 1.13), text(size: 7.5pt, fill: green.darken(20%))[a few hundred bytes,])
    content((11.3, 0.85), text(size: 7.5pt, fill: green.darken(20%))[answers percentile questions])
  }),
  caption: [
    Every position in the tensor sees its own stream of values across samples.
    We cannot store the streams — each position keeps a small summary instead.
  ],
) <fig-streams>

// ============================================================================
= Percentiles, in plain words
// ============================================================================

Line up everything a position has ever seen, smallest to largest. The
*median* (50th percentile) is the value halfway along the line: half of all
observations fall below it. The *95th percentile* is the value with 95% of
observations below it. Percentiles describe a distribution in the way people
actually use it: _typical value, normal range, extremes_ (@fig-quantile).

#figure(
  canvas(length: 1cm, {
    import draw: *
    // number line
    line((0, 0), (11, 0), stroke: 0.8pt + luma(120), mark: (end: ">", fill: luma(120)))
    content((11.2, -0.3), text(size: 8pt, fill: softgray)[value])
    // sorted sample dots (hand-picked, bell-ish)
    let vals = (1.1, 1.8, 2.3, 2.7, 3.0, 3.3, 3.5, 3.7, 3.9, 4.1, 4.3, 4.5, 4.7, 4.9, 5.2, 5.5, 5.8, 6.2, 6.7, 7.4, 9.0)
    for (i, v) in vals.enumerate() {
      let below = v <= 4.3
      circle((v, 0.35 + calc.rem(i * 7, 3) * 0.22), radius: 0.07,
        fill: if below { blue } else { blue.lighten(55%) }, stroke: none)
    }
    // median marker
    line((4.3, -0.15), (4.3, 1.45), stroke: (paint: navy, thickness: 1pt, dash: "dashed"))
    content((4.3, 1.75), text(size: 8pt, fill: navy)[*median (p50)* — half the dots sit below])
    // p95 marker
    line((7.4, -0.15), (7.4, 1.1), stroke: (paint: orange, thickness: 1pt, dash: "dashed"))
    content((8.35, 1.4), text(size: 8pt, fill: orange)[*p95* — only 1 in 20 lands beyond])
  }),
  caption: [
    Percentiles read a stream of values the way people do: "what is typical,
    and what is rare?" Dark dots are below the median.
  ],
) <fig-quantile>

The technical term for a percentile mark is a *quantile* — "the 0.95 quantile"
means the same as "the 95th percentile". We use the friendlier word from here
on.

== How it is done today: one t-digest per position

The classic tool for summarising a stream this way is the *t-digest*
#cite-mark(1). It compresses the stream into a few hundred little clusters,
keeping clusters small near the extremes (where precision matters most) and
letting them grow in the middle. It is a genuinely good algorithm, and it is
what monatq uses today — one t-digest per tensor position.

But per position it is expensive, and the expense is multiplied millions of
times:

- *Memory.* At the standard accuracy setting, each position reserves room for
  610 clusters — about *4.9 KB*. One million positions ≈ 5 GB just for
  summaries.
- *Work per update.* Deciding which clusters to merge is a chain of
  data-dependent decisions. Every position takes its own path, so the CPU
  cannot process positions in lockstep — a large hidden cost at tensor scale.
- *No guarantee.* Perhaps surprisingly, the t-digest's accuracy is empirical:
  there is no mathematical bound on how wrong it can be, and adversarial inputs
  that fool it are known.

The redesign below keeps what the t-digest gets right — precision focused on
the extremes — and removes the machinery.

// ============================================================================
= The idea: store the answers, not the machinery
// ============================================================================

Here is the whole idea in one sentence:

#align(center)[
  #block(inset: 8pt)[
    #text(size: 11pt, style: "italic", fill: navy)[
      Since we only ever ask for quantiles, store the quantiles themselves —
      a fixed set of marks — and keep them up to date as data streams in.
    ]
  ]
]

Each position keeps a *spine*: the values at $K$ fixed rank positions (for
example $K = 64$ marks: "the value 0.1% of the way up the sorted stream, the
value 0.5% of the way up, … the value 99.9% of the way up"). The marks are
placed *densely near the two ends* and sparsely in the middle — exactly where
precision is needed, the same insight that makes the t-digest good
(@fig-state).

Three small companions make the summary complete:

- *A records shelf.* The $T$ smallest and $T$ largest values ever seen, kept
  *exactly* (say $T = 8$ each), plus a short _recent-window_ shelf that
  restarts whenever the distribution visibly changes — so records can be
  dated: _all-time_ versus _since things last changed_. Record extremes are
  the one thing approximation should never touch — so they are not
  approximated at all.
- *A zero counter.* Neural networks produce huge numbers of _exact zeros_
  (a ReLU activation outputs zero about half the time). A smooth curve cannot
  represent a giant spike at one value, but a simple counter can — exactly.
- *A surprise gauge.* One small number tracking how much recent batches have
  disagreed with what the spine expected. It is computed for free inside the
  update (@sec-adapt) and is what lets the sketch _adapt_ when the stream
  changes instead of averaging the change away. Cost: 4 bytes.

#figure(
  canvas(length: 1cm, {
    import draw: *
    let W = 12.0
    let K = 16
    // ruler bar
    line((0, 0), (W, 0), stroke: 1.6pt + luma(100))
    content((0, -0.42), text(size: 7.5pt, fill: softgray)[rank 0%])
    content((W, -0.42), text(size: 7.5pt, fill: softgray)[rank 100%])
    content((W / 2, -0.42), text(size: 7.5pt, fill: softgray)[50%])
    // arcsine-spaced anchors: dense at the ends
    for j in range(0, K) {
      let q = 0.5 * (1.0 - calc.cos(calc.pi * j / (K - 1)))
      let x = W * q
      line((x, -0.09), (x, 0.09), stroke: 0.8pt + navy)
      circle((x, 0.32), radius: 0.07, fill: blue, stroke: none)
    }
    content((W / 2, 0.78), text(size: 8pt, fill: blue)[
      *the spine* — stored values at #box[$K$] fixed rank marks, crowded near the ends
    ])
    // brackets showing density
    content((0.62, -0.95), text(size: 7.5pt, fill: softgray)[marks crowd here…])
    content((11.3, -0.95), text(size: 7.5pt, fill: softgray)[…and here])
    line((0.6, -0.72), (0.6, -0.18), stroke: (paint: luma(150), thickness: 0.5pt))
    line((11.35, -0.72), (11.35, -0.18), stroke: (paint: luma(150), thickness: 0.5pt))

    // records shelves
    rect((-0.25, 1.3), (3.45, 2.15), fill: rgb("#ecfdf5"), stroke: 0.7pt + green, radius: 0.08)
    content((1.6, 1.92), text(size: 8pt, fill: green.darken(25%))[*records shelf (low)*])
    content((1.6, 1.55), text(size: 8pt, fill: green.darken(25%))[$T$ smallest values, exact])
    rect((8.55, 1.3), (12.25, 2.15), fill: rgb("#ecfdf5"), stroke: 0.7pt + green, radius: 0.08)
    content((10.4, 1.92), text(size: 8pt, fill: green.darken(25%))[*records shelf (high)*])
    content((10.4, 1.55), text(size: 8pt, fill: green.darken(25%))[$T$ largest values, exact])
    line((1.6, 1.28), (0.4, 0.5), stroke: (paint: green, thickness: 0.6pt, dash: "dotted"))
    line((10.4, 1.28), (11.6, 0.5), stroke: (paint: green, thickness: 0.6pt, dash: "dotted"))

    // zero counter
    rect((4.6, 1.3), (7.4, 2.15), fill: rgb("#fff7ed"), stroke: 0.7pt + orange, radius: 0.08)
    content((6.0, 1.92), text(size: 8pt, fill: orange.darken(15%))[*zero counter*])
    content((6.0, 1.55), text(size: 8pt, fill: orange.darken(15%))[exact count of 0-values])

    // byte budget
    content((6.0, -1.55), text(size: 8.5pt, fill: luma(70))[
      Total per position: #box[$K = 64$] anchors + records + counters + surprise gauge
      #h(4pt) → #h(4pt) #text(weight: "bold", fill: navy)[≈ 370 bytes] #h(6pt)
      (t-digest: ≈ 4 900 bytes)
    ])
  }),
  caption: [
    Everything one position stores. The spine holds values at fixed rank marks,
    dense near the extremes; the records shelves keep the true extremes exactly;
    the zero counter handles the ReLU spike. A few more bytes (not drawn) hold
    the windowed records, a promoted-spike counter, and the surprise gauge of
    @sec-adapt.
  ],
) <fig-state>

One more saving is almost free. Every position sees _the same number_ of
samples — the stream lengths are always identical, because each incoming tensor
contributes exactly one value to every position. So the sample count, and all
bookkeeping derived from it, is stored *once for the whole tensor* rather than
once per position. The t-digest cannot exploit this; its per-position cluster
weights must each carry their own counts.

// ============================================================================
= How it learns: blend, then re-read
// ============================================================================

New tensors arrive one at a time, but the spine does not update on every single
value. Instead, values are collected into a *batch* (a few hundred samples),
and the batch is folded in all at once. Batching is what monatq already does
for the t-digest — but for the spine the folding step becomes strikingly
simple:

+ *Sort* the batch for this position (its few hundred values).
+ *Update the records.* Compare the batch's smallest/largest values with the
  shelves; keep the best $T$ of each. Exact, and trivially cheap.
+ *Blend and re-read.* The spine describes the distribution of the $N$ values
  seen so far; the sorted batch describes the $B$ new ones _exactly_. Mix the
  two in proportion — $N$ parts old, $B$ parts new — and read off the values at
  the $K$ fixed rank marks again. Those become the new spine (@fig-blend).
+ *Note the surprise.* While sweeping, compare where the batch's values landed
  against where the spine expected them. The largest gap — the position's
  _surprise_ — costs one extra comparison per value and controls how the
  mixing weights are set (@sec-adapt): a stream behaving as usual blends
  gently; a stream that has visibly changed is absorbed quickly.

The blend is a weighted average of two curves, computed in a single sweep. There
are no clusters to manage, no decisions to make — every position performs the
same fixed arithmetic. This is what makes the update fast, and (as @sec-gains
explains) lets the CPU process eight positions at once.

#figure(
  canvas(length: 1cm, {
    import draw: *
    let W = 8.2       // x-axis length (value axis)
    let H = 3.4       // y-axis height (rank axis)
    let xmap(v) = W * (v / 6.0)
    // axes
    line((0, 0), (W + 0.4, 0), stroke: 0.7pt + luma(130), mark: (end: ">", fill: luma(130)))
    line((0, 0), (0, H + 0.3), stroke: 0.7pt + luma(130), mark: (end: ">", fill: luma(130)))
    content((W + 0.45, -0.3), text(size: 7.5pt, fill: softgray)[value])
    content((-0.15, H + 0.28), anchor: "east", text(size: 7.5pt, fill: softgray)[share of values below])
    content((-0.28, H), text(size: 7.5pt, fill: softgray)[100%])
    content((-0.28, 0), text(size: 7.5pt, fill: softgray)[0%])

    // old curve: normcdf((x-2.5)/0.85)
    let oldpts = ()
    for i in range(0, 121) {
      let v = i * 0.05
      oldpts.push((xmap(v), H * normcdf((v - 2.5) / 0.85)))
    }
    line(..oldpts, stroke: 1.1pt + blue)

    // batch ECDF: staircase of 12 values from a slightly shifted distribution
    let bvals = (1.9, 2.4, 2.8, 3.1, 3.35, 3.55, 3.75, 3.95, 4.15, 4.4, 4.7, 5.2)
    let n = bvals.len()
    let steps = ((xmap(1.4), 0),)
    for (k, v) in bvals.enumerate() {
      steps.push((xmap(v), H * k / n))
      steps.push((xmap(v), H * (k + 1) / n))
    }
    steps.push((xmap(5.8), H))
    line(..steps, stroke: 1.0pt + orange)

    // blended curve: 0.75 old + 0.25 batch (batch CDF approximated smoothly for drawing)
    let blendpts = ()
    for i in range(0, 121) {
      let v = i * 0.05
      let fold = normcdf((v - 2.5) / 0.85)
      let fbatch = normcdf((v - 3.55) / 0.78)
      blendpts.push((xmap(v), H * (0.75 * fold + 0.25 * fbatch)))
    }
    line(..blendpts, stroke: (paint: purple, thickness: 1.3pt, dash: "dashed"))

    // re-read anchors on the blended curve at fixed ranks
    for q in (0.05, 0.25, 0.5, 0.75, 0.95) {
      // find first blend point with y >= q*H
      let hit = blendpts.at(0)
      for p in blendpts {
        if p.at(1) < q * H { hit = p }
      }
      line((0, q * H), (hit.at(0), q * H), stroke: (paint: luma(170), thickness: 0.5pt, dash: "dotted"))
      circle((hit.at(0), q * H), radius: 0.085, fill: purple, stroke: white + 0.6pt)
    }

    // legend (white backing so it never collides with the curves)
    let lx = 0.5
    let ly = H + 0.05
    rect((lx - 0.2, ly - 1.42), (lx + 4.85, ly + 0.22), fill: rgb(255, 255, 255, 235), stroke: 0.4pt + luma(225), radius: 0.06)
    line((lx, ly), (lx + 0.55, ly), stroke: 1.1pt + blue)
    content((lx + 0.7, ly), anchor: "west", text(size: 7.5pt)[what the spine knew ($N$ values)])
    line((lx, ly - 0.4), (lx + 0.55, ly - 0.4), stroke: 1.0pt + orange)
    content((lx + 0.7, ly - 0.4), anchor: "west", text(size: 7.5pt)[new batch, known exactly ($B$ values)])
    line((lx, ly - 0.8), (lx + 0.55, ly - 0.8), stroke: (paint: purple, thickness: 1.3pt, dash: "dashed"))
    content((lx + 0.7, ly - 0.8), anchor: "west", text(size: 7.5pt)[blend — $N$ parts old, $B$ parts new])
    circle((lx + 0.28, ly - 1.2), radius: 0.085, fill: purple, stroke: white + 0.6pt)
    content((lx + 0.7, ly - 1.2), anchor: "west", text(size: 7.5pt)[re-read anchors → new spine])
  }),
  caption: [
    One update step. The rising curves show, for each value on the horizontal
    axis, what share of observations falls below it. The old knowledge (blue)
    and the exactly-known new batch (orange) are blended in proportion; the
    spine is then re-read at its fixed rank marks (purple dots).
  ],
) <fig-blend>

#deep([the update rule, formally])[
  Let $hat(F)_"old"$ be the distribution implied by the spine (interpolated
  through the anchors), $hat(F)_B$ the exact empirical distribution of the
  sorted batch, $N$ the running sample count and $B$ the batch size. The merged
  distribution is the mixture
  $ hat(F)_"new" = lambda hat(F)_"old" + (1 - lambda) hat(F)_B, quad lambda = N_"eff" / (N_"eff" + B), $
  where $N_"eff" = min(N, N_"max")$ caps the weight of history (fading memory)
  and $lambda$ is further reduced when the surprise gauge fires — both defined
  in @sec-adapt. The new anchors are the mixture's quantiles at the fixed ranks:
  $a'_j = hat(F)_"new"^(-1)(q_j)$. Because both inputs are monotone and sorted,
  this inversion is a single coordinated linear sweep costing $O(K + B)$ — no
  data-dependent branching. The only information lost anywhere in the pipeline
  is the interpolation between anchors of $hat(F)_"old"$; every other quantity
  in the rule is exact. The entire error analysis therefore reduces to
  controlling that one interpolation — which the refinements below make small.
]

// ============================================================================
= Why it stays accurate: four refinements
// ============================================================================

Reading a curve at fixed marks loses a little information each time. Done
naïvely, thousands of updates could let those little losses pile up. Four
refinements — each simple on its own — keep the total error tiny. Together they
are what turn the sketch from "plausible" into "sound".

== Wiggle the ruler (so errors cancel instead of piling up)

Each blend step commits a tiny reading error at each mark. The danger is not
the size of one error — it is that the _same_ error, repeated over
thousands of updates in the same direction, adds up in a straight line.

The fix is old statistical wisdom: *add a tiny random wiggle*. At each update,
the rank marks are shifted by a small random amount (well under the gap between
marks) before reading, then treated as the canonical marks. The reading errors
now point in a random direction each time — sometimes a hair high, sometimes a
hair low — and cancel out. After $M$ updates the accumulated error grows like
$sqrt(M)$ instead of $M$: after 10 000 updates, that is the difference between
*100× smaller error* and no improvement at all. The same trick, applied to a
different sketch, is what gave the well-known KLL algorithm #cite-mark(2) its
mathematical guarantee.

A second effect helps quietly: early errors fade. An error committed when the
sketch had seen $m$ batches is carried forward with weight proportional to
$m slash M$ — mistakes made when the sketch was young are mostly washed away by
later data.

== Use bell-curve graph paper (so the lines are nearly straight)

Between two marks, the spine has to guess by drawing a connecting line. How
much that guess loses depends entirely on how _curved_ the truth is between the
marks. So: change the paper.

Plotted on ordinary axes, the quantile curve of bell-shaped data bends hard at
both ends — exactly where straight-line guesses fail. But plotted against
*bell-curve-spaced axes* (the statistician's "probit scale"), the same curve
becomes a *perfectly straight line* whenever the data is bell-shaped
(@fig-link) — and neural-network tensors are usually close to bell-shaped.
A straight line between marks loses _nothing_; a nearly-straight one loses
almost nothing. The trick costs no memory: the axis spacing is a fixed table
shared by all million positions.

#figure(
  canvas(length: 1cm, {
    import draw: *
    let PW = 4.6      // panel width
    let PH = 3.2      // panel height
    let gap = 2.2
    let zmax = 2.6
    // anchors chosen by z so both panels share them
    let anchorz = (-2.33, -1.28, -0.52, 0.0, 0.52, 1.28, 2.33)

    // ---- Panel A: ordinary axes (x = rank q, y = value z) ----
    let ax(q) = PW * q
    let ay(z) = PH * (z + zmax) / (2 * zmax)
    rect((0, 0), (PW, PH), stroke: 0.5pt + luma(200))
    content((PW / 2, PH + 0.28), text(size: 8pt, fill: luma(70))[*ordinary graph paper*])
    content((PW / 2, -0.35), text(size: 7.5pt, fill: softgray)[rank (evenly spaced)])
    // true curve: (q, z) = (normcdf(z), z)
    let cpts = ()
    for i in range(0, 105) {
      let z = -zmax + i * (2 * zmax / 104)
      cpts.push((ax(normcdf(z)), ay(z)))
    }
    line(..cpts, stroke: 1.1pt + blue)
    // straight-line guesses between anchors
    let apts = ()
    for z in anchorz {
      apts.push((ax(normcdf(z)), ay(z)))
    }
    line(..apts, stroke: (paint: orange, thickness: 1pt, dash: "dashed"))
    for p in apts {
      circle(p, radius: 0.07, fill: navy, stroke: none)
    }
    content((PW * 0.62, PH * 0.16), text(size: 7.5pt, fill: orange.darken(10%))[gap = lost accuracy])
    line((PW * 0.40, PH * 0.175), (ax(normcdf(-1.75)) + 0.12, ay(-1.82)),
      stroke: 0.5pt + orange, mark: (end: ">", fill: orange))

    // ---- Panel B: bell-curve axes (x = probit of rank, y = value z) ----
    let ox = PW + gap
    let bx(z) = ox + PW * (z + zmax) / (2 * zmax)
    rect((ox, 0), (ox + PW, PH), stroke: 0.5pt + luma(200))
    content((ox + PW / 2, PH + 0.28), text(size: 8pt, fill: luma(70))[*bell-curve graph paper*])
    content((ox + PW / 2, -0.35), text(size: 7.5pt, fill: softgray)[rank (bell-curve spacing)])
    // the same curve is a straight line
    line((bx(-zmax), ay(-zmax)), (bx(zmax), ay(zmax)), stroke: 1.1pt + blue)
    for z in anchorz {
      circle((bx(z), ay(z)), radius: 0.07, fill: navy, stroke: none)
    }
    content((ox + PW * 0.52, PH * 0.16), text(size: 7.5pt, fill: green.darken(10%))[straight → nothing lost])
  }),
  caption: [
    The same bell-shaped distribution drawn twice. Left: on ordinary axes the
    quantile curve (blue) bends at the ends, and straight lines between anchors
    (orange, dashed) miss it. Right: on bell-curve-spaced axes the curve _is_
    a straight line — connecting the same anchors is exact.
  ],
) <fig-link>

For data that is bell-shaped after taking logarithms (also common in neural
networks), the same idea applies with a log transform first. Where extra
smoothness is wanted, the straight connecting lines can be upgraded to gentle
curves (monotone cubics #cite-mark(3)) — a standard, safe interpolation that
never invents spurious wiggles.

== Count the zeros separately

About half of all values after a ReLU activation are _exactly zero_ — a giant
spike at a single point (@fig-atom). Any method built on smooth curves — the
t-digest included — smears such a spike out. The spine sidesteps the problem:
a per-position counter records the zeros exactly, and the spine models only the
nonzero values, which are smooth and well-behaved. At query time the two parts
are recombined exactly. Cost: 4 bytes. (Spikes at _other_, unanticipated
values are caught at run time and promoted to counters of their own — see
@sec-adapt.)

#figure(
  canvas(length: 1cm, {
    import draw: *
    let bw = 0.42
    // axis
    line((-0.4, 0), (8.6, 0), stroke: 0.7pt + luma(130), mark: (end: ">", fill: luma(130)))
    content((8.7, -0.3), text(size: 7.5pt, fill: softgray)[value])
    // zero spike
    rect((0, 0), (bw, 3.1), fill: orange.lighten(20%), stroke: 0.5pt + orange.darken(15%))
    content((bw / 2, -0.32), text(size: 7.5pt, fill: softgray)[0])
    content((4.85, 3.18), text(size: 8pt, fill: orange.darken(15%))[*≈ half of everything is exactly 0*])
    content((4.85, 2.86), text(size: 7.5pt, fill: softgray)[one counter, exact])
    line((2.5, 3.18), (0.55, 3.12), stroke: 0.5pt + orange, mark: (end: ">", fill: orange))
    // smooth positive part: half-bell bars
    let heights = (1.55, 1.75, 1.62, 1.38, 1.1, 0.82, 0.58, 0.38, 0.24, 0.14, 0.08, 0.045, 0.025)
    for (i, h) in heights.enumerate() {
      let x = 0.75 + i * (bw + 0.09)
      rect((x, 0), (x + bw, h), fill: blue.lighten(45%), stroke: 0.5pt + blue.darken(5%))
    }
    content((5.35, 1.78), text(size: 8pt, fill: navy)[smooth remainder — handled by the spine])
    line((3.55, 1.68), (2.65, 1.2), stroke: 0.5pt + navy, mark: (end: ">", fill: navy))
  }),
  caption: [
    Values at a position after a ReLU activation. A smooth curve cannot honestly
    represent the spike at zero; a counter represents it perfectly.
  ],
) <fig-atom>

== Be exact whenever exactness is free

Two moments in a sketch's life allow exactness at zero cost, and the spine
takes both:

- *Early life.* Until a position has seen more values than it has anchors
  ($N <= K$), the sorted values simply _are_ the state — nothing is
  approximated. Blending only begins once the stream outgrows the storage.
- *The extremes, always.* The records shelves keep the $T$ true smallest and
  largest values at all times. Questions about the far tail — "the worst value
  in a hundred thousand" — are answered from the shelf, _exactly_. The
  t-digest, by contrast, always interpolates near the extremes. For the deep
  region between the shelf and the outermost anchor, a classical
  extreme-value formula #cite-mark(4) can be fitted _at query time_, using no
  stored memory at all: queries are rare, so they can afford to think.

#deep([the accuracy picture in numbers])[
  Let $Delta$ be the gap between neighbouring rank marks (on the transformed
  axis) and $M = N slash B$ the number of blend steps. One blend's reading
  error at a mark is $O(Delta^2)$ for straight-line interpolation — and
  vanishes entirely for bell-shaped data on the probit axis; monotone cubics
  reduce the generic case to $O(Delta^4)$. With random dithering the errors
  form a zero-mean sequence with fading influence ($m slash M$ for the error
  from blend $m$), so the accumulated error concentrates at
  $O(Delta^2 sqrt(M slash 3))$ (Azuma), rather than the $O(Delta^2 M)$ of the
  naive scheme. (Fading memory, @sec-adapt, additionally caps $M$ at
  $N_"max" slash B$, making the bound uniform over arbitrarily long streams.) Balancing interpolation error against the unavoidable
  statistical noise of $N$ samples (quantile CLT: standard error
  $prop 1 slash sqrt(N)$) shows there is no benefit in growing $K$ beyond
  $Theta(N^(1 slash 4))$ — for $N = 10^6$, a spine of $K approx 64$ marks
  already sits at the statistical noise floor. That is the formal justification
  for "64 numbers are enough".
]

// ============================================================================
= How it adapts: surprise as a signal <sec-adapt>
// ============================================================================

The refinements above assume the stream keeps behaving like itself. Real
streams are not always so polite: a distribution can drift as data changes, a
regime can flip outright (a new dataset, a model edit), and a spike can hide
at a value nobody anticipated. Rather than hoping such surprises never happen,
the spine _measures_ them — and the measurement turns out to be nearly free.

== A surprise gauge, free of charge

The blend step already walks the sorted batch against the spine's belief. One
extra comparison per value records the largest disagreement between where the
batch's values landed and where the spine expected them — a single number $D$
per position, the position's *surprise*. Statistics says exactly how large $D$
should be when nothing has changed (it shrinks like one over the square root
of the batch size), so "surprising" has a principled threshold rather than a
tuned knob. The gauge is 4 bytes of state; the extra work hides inside the
sweep the spine was doing anyway. It is also a useful output in its own right:
a per-position *surprise map* shows exactly which neurons changed behaviour.

== Memory that fades, a gain that reacts

The plain blend rule — $N$ parts old, $B$ parts new — never forgets. After a
million samples, a batch of five hundred moves the spine by less than a
twentieth of a percent, so a drifting stream is chased with ever-growing lag.
Two changes fix this:

- *Cap the memory.* Blend as if the spine had seen at most $N_"max"$ values.
  Old evidence then fades geometrically, with an explicit, chosen time
  constant — and the accuracy guarantee survives; in fact it becomes
  _uniform_: the error bound no longer grows with stream length at all.
- *Let surprise open the gate.* When the gauge fires, the blend tilts further
  toward the new batch — the more surprising, the stronger the tilt, in the
  spirit of a Kalman gain. A genuinely shifted stream rewrites the spine
  within a few batches instead of thousands (@fig-adapt); ordinary noise,
  below the threshold, changes nothing at all.

#figure(
  canvas(length: 1cm, {
    import draw: *
    let W = 11.4
    let H = 3.1
    let xmap(t) = W * t / 100
    let ymap(v) = H * (v - 1.2) / 3.4
    // shaded detection-lag band (under everything)
    rect((xmap(50), 0), (xmap(56), H * 0.97), fill: rgb("#f3e8ff"), stroke: none)
    // axes
    line((0, 0), (W + 0.4, 0), stroke: 0.7pt + luma(130), mark: (end: ">", fill: luma(130)))
    line((0, 0), (0, H + 0.3), stroke: 0.7pt + luma(130), mark: (end: ">", fill: luma(130)))
    content((W + 0.5, -0.3), text(size: 7.5pt, fill: softgray)[batches])
    content((-0.15, H + 0.28), anchor: "east", text(size: 7.5pt, fill: softgray)[median at one position])
    content((xmap(50), -0.32), text(size: 7.5pt, fill: softgray)[change point])
    // true behaviour: step (dashed gray)
    line((xmap(1), ymap(2.0)), (xmap(50), ymap(2.0)), stroke: (paint: luma(120), thickness: 1pt, dash: "dashed"))
    line((xmap(50), ymap(2.0)), (xmap(50), ymap(4.0)), stroke: (paint: luma(120), thickness: 1pt, dash: "dashed"))
    line((xmap(50), ymap(4.0)), (xmap(100), ymap(4.0)), stroke: (paint: luma(120), thickness: 1pt, dash: "dashed"))
    // never-forgetting blend
    let naive = ()
    for t in range(1, 101) {
      let v = if t <= 50 { 2.0 } else { (50.0 * 2.0 + (t - 50) * 4.0) / t }
      naive.push((xmap(t), ymap(v)))
    }
    line(..naive, stroke: 1.1pt + orange)
    // adaptive spine
    let adap = ()
    for t in range(1, 101) {
      let v = if t <= 52 { 2.0 } else if t <= 55 { 2.0 + (t - 52) * (2.0 / 3.0) } else { 4.0 }
      adap.push((xmap(t), ymap(v)))
    }
    line(..adap, stroke: 1.3pt + purple)
    // annotation for the band
    content((xmap(76), 0.62), text(size: 7.5pt, fill: purple.darken(10%))[gauge fires → intervals widen briefly])
    line((xmap(63.5), 0.62), (xmap(55), 0.95), stroke: 0.5pt + purple, mark: (end: ">", fill: purple))
    // legend
    let lx = 0.5
    let ly = 2.62
    rect((lx - 0.2, ly - 1.06), (lx + 4.65, ly + 0.24), fill: rgb(255, 255, 255, 235), stroke: 0.4pt + luma(225), radius: 0.06)
    line((lx, ly), (lx + 0.55, ly), stroke: (paint: luma(120), thickness: 1pt, dash: "dashed"))
    content((lx + 0.7, ly), anchor: "west", text(size: 7.5pt)[what the stream actually does])
    line((lx, ly - 0.4), (lx + 0.55, ly - 0.4), stroke: 1.1pt + orange)
    content((lx + 0.7, ly - 0.4), anchor: "west", text(size: 7.5pt)[blend that never forgets — lags])
    line((lx, ly - 0.8), (lx + 0.55, ly - 0.8), stroke: 1.3pt + purple)
    content((lx + 0.7, ly - 0.8), anchor: "west", text(size: 7.5pt)[adaptive spine — snaps quickly])
  }),
  caption: [
    A regime change, seen by one position. The never-forgetting blend (orange)
    closes the gap only in proportion to elapsed stream length — fifty batches
    later it is still far off. The adaptive spine (purple) notices the change
    on its surprise gauge, restarts, and snaps to the new behaviour within a
    few batches, reporting widened intervals during the brief lag (shaded).
  ],
) <fig-adapt>

== When the world truly changes: restart, and a second shelf

Sustained surprise — several loud batches in a row — means the old
distribution is simply gone. A tiny two-bit state machine per position
(_calm → alert → restart_) makes the response explicit. On restart, the spine
keeps its anchors as a warm starting guess but drops their weight, so the next
few batches dominate. The restart is expressed entirely through the state
bits, so the sample count stays a single global number — the shared-stream
saving of the base design is preserved.

Restarts also solve a quiet problem with records: an all-time record is
forever, and after a regime change it describes a distribution that no longer
exists. This is what the *windowed shelf* is for — the extremes _since the
last change-point_, cleared at each restart. Queries can answer both "worst
ever" and "worst lately", and say which is which.

== Hidden spikes become counters

The zero counter handles the one spike we can predict. But spikes appear at
other values too: saturation limits, clamped activations, a dead neuron stuck
at a constant. The sorted batch betrays them immediately — a spike is a run of
_tied values_, and ties are counted during the sweep at no extra cost. A value
that keeps showing up heavily is *promoted*: it gets its own exact counter,
just like zero, and the spine models only the smooth remainder. The worst case
for the interpolation assumption is thereby converted into the best case — an
exactly counted atom. Cost: 8 bytes.

== The ruler adapts too — for the whole tensor, occasionally

Per-position changes to the rank grid or the axis transform would break the
lockstep property that makes updates fast. But the grid and the transform are
_shared_, so they can adapt at the tensor level, where the cost is trivial.
Every so often, the library checks which axis choice (bell-curve,
log-then-bell-curve, …) makes the pooled tensor's anchors straightest, and
whether one region of rank space is doing most of the bending. If so, the
shared ruler is re-chosen and every spine is re-read through it once — one
extra sweep, amortized to nothing. Every position still performs identical
arithmetic; adaptivity lives entirely in shared metadata.

== Queries confess

Finally, surprise flows into answers. The query-time confidence intervals
widen with the recent gauge reading and shrink again as the sketch
re-converges after a restart: a shifting position reports "p99 = 4.1 ± 0.9,
distribution changing" rather than a crisply wrong number. No failure mode of
the adaptation machinery is silent — at worst, the sketch tells you it is
unsure.

#deep([the adaptation rules, formally])[
  *Gauge.* During the blend sweep,
  $D = max_k |tilde(F)_"old" (b_((k))) - k slash B|$ — the Kolmogorov–Smirnov
  distance between the spine's belief and the exact batch. If the stream is
  unchanged, the DKW inequality bounds it:
  $Pr[D > sqrt(ln(2 slash beta) slash (2 B))] <= beta$ — a principled alert
  threshold $tau$ with no tuned constants. The stored gauge is an exponential
  average, $macron(D) <- (1 - rho) macron(D) + rho D$, quantized into the
  4-byte surprise word alongside the 2-bit regime state.

  *Gain.* The blend weight becomes
  $lambda = gamma dot N_"eff" slash (N_"eff" + B)$ with
  $N_"eff" = min(N, N_"max")$ and $gamma = exp(-c B max(0, D - tau)^2)$,
  activated only once $D > 1.5 tau$ (below that, $gamma = 1$). The hysteresis
  band was added after implementation: without it, rare stationary batches
  graze the threshold and trigger false restarts. Above the band, a strongly
  surprising batch drives $gamma$ toward zero and the spine toward the batch.

  *Guarantee.* With $N_"eff"$ capped, the influence of the blend-$m$ error
  decays geometrically rather than as $m slash M$, and the Azuma argument
  yields an accumulated-error bound $O(w sqrt(N_"max" slash B))$ that is
  _uniform in stream length_ — drift-tracking costs a constant, not a new
  failure mode. A regime shift exceeding $tau$ in KS distance is detected
  after $O(1)$ batches.
]

// ============================================================================
= What you gain <sec-gains>
// ============================================================================

*Memory.* About *13× less* per position (@fig-memory). A summary of a
million-position tensor drops from ≈ 4.9 GB to ≈ 0.4 GB — the difference
between "needs a big server" and "runs on a laptop".

#figure(
  canvas(length: 1cm, {
    import draw: *
    let scale = 1.9   // cm per KB
    let bh = 0.62
    // t-digest bar
    rect((0, 1.1), (4.9 * scale, 1.1 + bh), fill: orange.lighten(25%), stroke: 0.6pt + orange.darken(10%))
    content((0, 1.1 + bh + 0.26), anchor: "west", text(size: 8.5pt)[*t-digest* (accuracy setting 100)])
    content((4.9 * scale + 0.15, 1.1 + bh / 2), anchor: "west", text(size: 8.5pt, fill: orange.darken(20%))[*≈ 4 900 B / position*])
    // spine bar
    rect((0, 0), (0.368 * scale, bh), fill: blue.lighten(25%), stroke: 0.6pt + navy)
    content((0, bh + 0.26), anchor: "west", text(size: 8.5pt)[*Quantile Spine* ($K = 64$, $T = 8$, $T_w = 4$)])
    content((0.368 * scale + 0.15, bh / 2), anchor: "west", text(size: 8.5pt, fill: navy)[*≈ 370 B / position — 13× smaller*])
  }),
  caption: [Summary memory per tensor position, drawn to scale.],
) <fig-memory>

*Update speed.* The spine's update is the same fixed arithmetic at every
position — sort a small batch, sweep, write $K$ numbers. Modern CPUs have
*SIMD* instructions that apply one operation to eight numbers at once, but only
when all eight follow the same path. The t-digest's per-position decision
making forbids this; the spine's uniformity invites it. Better still, monatq's
existing memory layout already stores the batch values of neighbouring
positions side by side, so eight positions can be sorted and swept _in
lockstep_ using branch-free sorting networks. The most expensive step of the
whole pipeline parallelises eight-wide essentially for free.

*Query speed and flexibility.* Reading a quantile is a lookup between two
anchors — no cluster arithmetic. Combining positions (for example, pooling a
whole channel into one distribution, monatq's `merge_cells`) becomes a simple
weighted average of spines rather than the t-digest's collect-sort-recompress
dance. And because queries are rare, they can afford luxuries updates cannot:
extreme-value tail fitting, and even confidence intervals — "the p99 is 4.1 ±
0.2" — which the t-digest cannot provide at all. The intervals are
surprise-aware (@sec-adapt): while the gauge reports recent change they widen
automatically, so a shifting stream is never answered with a crisply wrong
number.

== Side by side

#figure(
  block(width: 100%)[
    #set text(size: 8.8pt)
    #table(
      columns: (1.35fr, 1.55fr, 1.55fr),
      align: (left, left, left),
      stroke: 0.4pt + luma(200),
      inset: 6pt,
      fill: (x, y) => if y == 0 { rgb("#eef2ff") } else if calc.odd(y) { luma(250) } else { white },
      [*Property*], [*t-digest (per position)*], [*Quantile Spine*],
      [Memory per position], [≈ 4 900 B], [*≈ 370 B (≈ 13× less)*],
      [Update work], [sort + adaptive cluster management (branchy)], [*sort + one fixed-cost sweep*],
      [Eight positions at once (SIMD)], [no — each takes its own path], [*yes — identical lockstep arithmetic*],
      [Record extremes], [approximated (interpolated)], [*exact* ($T$ true records kept)],
      [Spikes (e.g. ReLU zeros)], [smeared into clusters], [*exact* (dedicated counter)],
      [Sample-count bookkeeping], [per cluster, per position], [*once, globally* (shared streams)],
      [Pooling positions together], [collect + sort + recompress], [*weighted average of curves*],
      [Confidence intervals], [not available], [*at query time, surprise-aware*],
      [Accuracy guarantee], [none (empirical only)], [*concentration bound under mild assumptions*],
      [Distribution drift], [all history weighted equally — lag grows forever], [*fading memory + change-point restarts*],
      [Surprise detection], [not available], [*per-position gauge, free in the update sweep*],
      [Distribution assumptions], [none], [mild (bell-ish between marks) — *monitored, adapted when violated*],
      [Track record], [a decade in production use], [young — but *validated head-to-head* (@sec-results)],
    )
  ],
  caption: [Per-position t-digest versus Quantile Spine, for the tensor-tracking workload.],
) <tbl-compare>

// ============================================================================
= Honest trade-offs
// ============================================================================

No design is free, and two costs deserve plain statement.

*Adaptation reacts; it does not predict.* A genuine change in the stream is
absorbed only after it registers on the surprise gauge — a lag of a batch or
two, during which queries answer with honestly widened intervals rather than
fresh values. The gauge's thresholds also assume batches are informative
samples of current behaviour; a stream _ordered by an adversary_ can still
slow the machinery down. (The t-digest offers no adversarial guarantee either;
the spine's assumption is explicit and, unlike the t-digest's, monitored.)

*The ruler adapts per tensor, not per position.* $K$, $T$, the link and the
grid are shared by all positions so that updates stay in lockstep; the refit
of @sec-adapt moves them for the tensor as a whole. A lone position whose
shape differs wildly from its tensor is protected by its records, its
promoted atoms, and widened intervals — but its mid-range interpolation is
only as good as the shared ruler allows. The deepest tail region between the
outermost anchor and the records shelf is likewise covered by the query-time
extreme-value fit rather than by stored anchors.

And one meta-cost remains: the t-digest has a decade of production hardening;
the spine's implementation is new. The head-to-head measurement the first
edition of this paper called for has since been carried out — @sec-results
reports it — but breadth of real-world exposure only time can provide.

// ============================================================================
= Measured: the design holds up <sec-results>
// ============================================================================

The first edition of this paper ended by calling for a head-to-head
measurement. It has since been carried out: the full design — flat
per-position arrays, parallel flushes, the eight-wide lockstep sweep of
@sec-gains, and the shared-ruler refit of @sec-adapt — is implemented in
monatq and compared against monatq's production t-digest (compression 100) on
identical streams, with the exact sorted values as referee. All
numbers below are from an Apple M4 (10 cores).

== Speed

#figure(
  block(width: 100%)[
    #set text(size: 8.8pt)
    #table(
      columns: (1.9fr, 1fr, 1.1fr, 0.8fr),
      align: (left, right, right, right),
      stroke: 0.4pt + luma(200),
      inset: 6pt,
      fill: (x, y) => if y == 0 { rgb("#eef2ff") } else if calc.odd(y) { luma(250) } else { white },
      [*Workload*], [*t-digest*], [*Quantile Spine*], [*speedup*],
      [ingest — 64×64 tensor, 1 000 samples (normal)], [536 M/s], [*666 M/s*], [1.2×],
      [ingest — 64×64 tensor, 1 000 samples (uniform)], [536 M/s], [*657 M/s*], [1.2×],
      [ingest — 256×256 tensor, 200 samples (uniform)], [472 M/s], [*877 M/s*], [1.9×],
      [query — p99 at every position, 64×64], [53 M/s], [*277 M/s*], [5.2×],
    )
  ],
  caption: [Measured throughput (values per second, divan medians).],
) <tbl-speed>

Three implementation lessons are worth recording, all now folded into the
design. First, a straightforward scalar version of the spine was about 3×
_slower_ than the t-digest on small tensors: uniform arithmetic only pays once
the eight-wide lockstep sweep is actually used — transposed tiles, branch-free
sorting networks, a vectorised gauge, and an $O(K)$ fast path for calm, smooth
batches. Second, computing the surprise gauge naively consumed half of every
flush; screening the KS maximum against the anchor grid, with an exact
fallback only near threshold crossings, made it nearly free — as the design
promised, but only after the screening was added. Third, threading the
refit's link choice through the query path initially cost queries 25–40% —
recovered, and then some, by resolving the link once per query and caching
the cubic tangents: queries ended up *four times faster* than before the
refit work began.

== Accuracy

#figure(
  block(width: 100%)[
    #set text(size: 8.8pt)
    #table(
      columns: (1.4fr, 1fr, 1fr, 1fr, 1fr),
      align: (left, right, right, right, right),
      stroke: 0.4pt + luma(200),
      inset: 6pt,
      fill: (x, y) => if y == 0 { rgb("#eef2ff") } else if calc.odd(y) { luma(250) } else { white },
      [*Distribution*], [*t-digest mean*], [*t-digest worst*], [*Spine mean*], [*Spine worst*],
      [normal], [0.00063], [0.00695], [*0.00030*], [*0.00149*],
      [uniform], [0.00133], [0.00738], [*0.00058*], [*0.00178*],
      [lognormal], [0.00183], [0.01257], [*0.00076*], [*0.00213*],
      [ReLU-like (50% zeros)], [0.02703], [0.25223], [*0.00009*], [*0.00057*],
    )
  ],
  caption: [
    Rank error — how far the returned value's true rank sits from the
    requested one — over nine quantiles from 0.001 to 0.999, $N = 100 000$
    samples per position, exact sorted ground truth.
  ],
) <tbl-acc>

The worst-case column is where tails dominate — and where the query-time
monotone cubics and generalized-Pareto tail fit collapse the spine's error to
a flat profile: no quantile, tail or middle, strays beyond 0.0022 in rank.
The dithering prediction was checked directly: on a stationary stream the
anchor error _shrinks_ as blends accumulate — the $sqrt(M)$ cancellation, with
no linear drift. An earlier round of these measurements left the t-digest one
lead — _mean_ error on uniform data, the link-mismatch case. Implementing the
shared-ruler refit of @sec-adapt erased it: the refit selects the linear axis
for uniform streams, where straight-line interpolation is near-exact, and the
uniform mean error fell from 0.00184 to 0.00058 — better than twice the
t-digest's. A related correction — blending the calm-path batch with expected
order statistics under the matching link — removed a subtle finite-batch
shrink of the tails, improving the normal and lognormal rows as well. Every
row of the table is now won by the spine, on both columns.

== Adaptation and memory

#figure(
  block(width: 100%)[
    #set text(size: 8.8pt)
    #table(
      columns: (1.4fr, 1fr, 1fr),
      align: (left, right, right),
      stroke: 0.4pt + luma(200),
      inset: 6pt,
      fill: (x, y) => if y == 0 { rgb("#eef2ff") } else if calc.odd(y) { luma(250) } else { white },
      [*Samples after the change*], [*t-digest error*], [*Spine error*],
      [256], [3.998], [*0.117*],
      [4 096], [3.904], [*0.060*],
      [50 000], [1.947], [*0.029*],
    )
  ],
  caption: [
    Mean absolute error of the reported median after an abrupt
    $N(0,1) -> N(4,1)$ regime change (the true median moves by 4).
  ],
) <tbl-drift>

Fifty thousand samples after the change, the never-forgetting t-digest still
reports a median off by _half the shift_; the spine is within 0.03 of the
truth a few hundred samples in — the fading memory and surprise gain doing
exactly what @sec-adapt claimed. The memory claim held to the byte: *368 bytes
per position* measured, against 4 900 — the predicted 13.3×.

// ============================================================================
= Summary
// ============================================================================

Quantile Spine replaces "store a machine that can compute quantiles" with
"store the quantiles". A fixed spine of rank marks — dense at the extremes,
straightened by bell-curve axes, kept honest by a random wiggle — plus exact
records, exact counters for spikes, and a surprise gauge that turns
distribution change from a failure mode into a tracked signal, answers the
same questions as a per-position t-digest in about *13× less memory*, with
updates that run in lockstep across positions, exact extremes, and a form of
accuracy statement the t-digest cannot make. Its assumptions are mild,
explicit, monitored at run time, and tailored to what tensor streams actually
look like. And these are no longer paper claims: the implemented sketch
matches or beats the t-digest on every measured axis — memory, update and
query speed, worst-case accuracy, spikes, and drift (@sec-results).

#v(1em)
#line(length: 100%, stroke: 0.6pt + luma(200))

// ============================================================================
// Appendix: formal specification
// ============================================================================
#heading(numbering: none)[Appendix: formal specification]
#set text(size: 9pt)

*State* (per position): anchors $a_1 <= dots <= a_K$; exact order statistics
$"lo"_(1..T)$, $"hi"_(1..T)$ (all-time) and $"lo"^w_(1..T_w)$,
$"hi"^w_(1..T_w)$ (since last change-point); atom counters $n_0$ (zero) and
$(v_1, n_1)$ (promoted secondary atom, if any); surprise word: quantized gauge
$macron(D)$ plus 2-bit regime state. Shared globally: sample count $N$, memory
cap $N_"max"$, batch size $B$, link $g$, rank grid and its transformed values.

*Rank grid.* Arcsine-spaced ranks $q_j = 1/2 (1 - cos(pi j slash (K - 1)))$,
$j = 0, dots, K - 1$ (equidistributed under the t-digest $k_1$ scale
function), or uniform in probit space $z_j = Phi^(-1)(q_j)$ on a clipped range.

*Interpolant.* $tilde(Q)(q) = cal(I)((z_j, a_j)_j; g(q))$ with link
$g = Phi^(-1)$ (identity and log-probit $Phi^(-1) compose ln$ are the
alternatives, chosen by the shared-ruler refit) and $cal(I)$ monotone
piecewise-linear or Fritsch–Carlson cubic. The implied CDF $tilde(F)$ is the
inverse of $tilde(Q)$, combined with the atom mass $n_0 slash N$ at $0$ and the
exact tail segments.

*Update* (per flushed batch, per position): sort batch ($O(B log B)$,
lane-parallel); merge shelves ($O(T + T_w)$); count atoms and adjacent ties
(modal tie fraction $>= theta$ over successive batches promotes $(v_1, n_1)$);
compute $D$ in the same sweep (below); re-anchor
$a'_j = (lambda tilde(F)_"old" + (1 - lambda) hat(F)_B)^(-1)(q_(j + eta))$,
$lambda = gamma dot N_"eff" slash (N_"eff" + B)$,
$N_"eff" = min(N, N_"max")$, $gamma = exp(-c B max(0, D - tau)^2)$ for
$D > 1.5 tau$ (else $gamma = 1$; hysteresis against stationary false alerts),
with shared dither $eta ~ "Unif"(-1/2, 1/2)$ applied to the grid index — one
$O(K + B)$ sweep, eight positions per SIMD lane group, with an $O(K)$
quantile-space fast path for calm smooth batches.

*Surprise and regimes.* $D = max_k |tilde(F)_"old" (b_((k))) - k slash B|$,
screened against the anchor grid with an exact fallback near threshold
crossings and atoms (keeps the gauge at a small fraction of flush cost);
under no change, DKW gives $Pr[D > tau_beta] <= beta$ for
$tau_beta = sqrt(ln(2 slash beta) slash (2 B))$. Gauge
$macron(D) <- (1 - rho) macron(D) + rho D$. Regime bits: calm → alert on one
crossing; alert → restart on $R$ consecutive crossings. Restart clears the
windowed shelves and suppresses $lambda$ for the next $R'$ batches, encoded in
the state bits — no per-position sample count is stored. Query-time intervals
inflate by a term monotone in $macron(D)$.

*Shared-ruler refit* (per tensor, every $E$ flushes; implemented with
$E = 16$): candidate links $g = Phi^(-1)$ (probit), identity, or
$Phi^(-1) compose ln$ (log-probit, admitted only when all anchors and shelf
minima are positive); straightness scored as normalized least-squares residual
energy of location–scale-normalized pooled anchor curves (up to 256 positions
pooled) against a straight line; 20% hysteresis before switching; forced refit
after early-life exit and after a regime restart. A link switch is
metadata-only — anchors are values at fixed ranks and do not change; grid
re-warping proved unnecessary in practice. Amortized $O(K slash E)$ per
position per flush; positions remain in lockstep — adaptivity lives only in
shared metadata.

*Error.* Per-merge resampling error $w = O(Delta z^2)$ (linear; $0$ for exact
location–scale data under the matching link) or $O(Delta z^4)$ (cubic). The
influence of the blend-$m$ error on the final state scales as $m slash M$
(mixture-weight telescoping, $M = N slash B$ blends). Under the dithering
assumption (conditionally zero-mean increments), Azuma's inequality gives, with
probability $>= 1 - beta$, accumulated anchor error
$<= w sqrt((2 M slash 3) ln (2 slash beta))$ — versus $Theta(M w)$
deterministically without dithering. Balancing $w$ against the quantile CLT
noise floor $sqrt(q (1 - q) slash N) slash f(Q(q))$ yields the matched
resolution $K^* = Theta(N^(1 slash 4))$ (linear) or $Theta(N^(1 slash 8))$
(cubic). With fading memory ($N_"eff" <= N_"max"$), blend influences decay
geometrically and the accumulated bound tightens to
$O(w sqrt(N_"max" slash B))$ — uniform in stream length.

*Memory.* $4 (K + 2 T + 2 T_w + 4)$ bytes per position ($K = 64$, $T = 8$,
$T_w = 4$: 368 B — anchors, both shelves, zero and atom counters, surprise
word), against $8 (6 delta + 10) + 16$ bytes for the t-digest bound
($delta = 100$: ≈ 4.9 KB).

#v(0.6em)
#heading(numbering: none)[References]

#set enum(numbering: "[1]")
+ T. Dunning, O. Ertl. _Computing extremely accurate quantiles using
  t-digests._ arXiv:1902.04023, 2019.
+ Z. Karnin, K. Lang, E. Liberty. _Optimal quantile approximation in
  streams._ FOCS 2016. (The "KLL" sketch; source of the randomized-compaction
  idea echoed by the spine's dithering.)
+ F. N. Fritsch, R. E. Carlson. _Monotone piecewise cubic interpolation._
  SIAM J. Numer. Anal. 17(2), 1980.
+ J. Pickands. _Statistical inference using extreme order statistics._
  Ann. Statist. 3(1), 1975; A. Balkema, L. de Haan. _Residual life time at
  great age._ Ann. Probab. 2(5), 1974. (Basis of generalized-Pareto tail
  fitting.)
+ E. Gan, J. Ding, K. S. Tai, V. Sharan, P. Bailis. _Moment-based quantile
  sketches for efficient high-cardinality aggregation queries._ VLDB 2018.
