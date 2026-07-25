#import "@preview/cetz:0.4.2"

#let navy = rgb("17324d")
#let blue = rgb("177ddc")
#let cyan = rgb("39a9c6")
#let orange = rgb("e67e22")
#let red = rgb("c83e4d")
#let green = rgb("2a9d6f")
#let purple = rgb("7656a7")
#let ink = rgb("17212b")
#let muted = rgb("617282")
#let gridline = rgb("dbe5eb")
#let pale = rgb("f4f7f9")

#set page(
  paper: "us-letter",
  margin: (top: 0.70in, bottom: 0.70in, left: 0.76in, right: 0.76in),
  header: context {
    if counter(page).get().first() > 1 [
      #set text(size: 8pt, fill: muted)
      RANKSTORE · compact tensor histories for deferred analysis
      #h(1fr)
      Research note · version 0.3
    ]
  },
  footer: context {
    if counter(page).get().first() > 1 [
      #set text(size: 8pt, fill: muted)
      Compact observation proposal — Rust throughput remains unmeasured
      #h(1fr)
      #counter(page).display("1")
    ]
  },
)
#set text(font: ("New Computer Modern", "Libertinus Serif"), size: 9.4pt, fill: ink)
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.1", outlined: true)
#show heading.where(level: 1): it => {
  v(0.7em)
  set text(size: 18pt, weight: "bold", fill: navy)
  it
  v(0.2em)
  line(length: 100%, stroke: 0.8pt + blue)
  v(0.35em)
}
#show heading.where(level: 2): set text(size: 13pt, weight: "bold", fill: navy)
#show heading.where(level: 3): set text(size: 10.5pt, weight: "bold", fill: navy)
#show link: set text(fill: blue)
#set table(stroke: 0.45pt + gridline, inset: 5pt)
#set figure(gap: 0.4em)
#show figure.caption: set text(size: 8.3pt, fill: muted)

#let callout(title, body, tone: blue) = block(
  width: 100%,
  fill: tone.lighten(91%),
  stroke: (left: 3pt + tone),
  inset: (x: 11pt, y: 8pt),
  radius: 3pt,
  [#text(weight: "bold", fill: tone)[#title] #h(4pt) #body],
)

#let pill(body, tone: blue) = box(
  fill: tone.lighten(88%),
  radius: 9pt,
  inset: (x: 7pt, y: 3pt),
  text(size: 8pt, weight: "bold", fill: tone, body),
)

#let intuition(title, equation, art, note) = figure(
  grid(
    columns: (0.43fr, 0.57fr),
    gutter: 10pt,
    block(
      fill: pale,
      stroke: 0.55pt + gridline,
      radius: 4pt,
      inset: 10pt,
      [
        #text(size: 8pt, weight: "bold", fill: muted)[THE FORMULA]
        #v(0.45em)
        #align(center, text(size: 11pt, equation))
      ],
    ),
    block(
      fill: blue.lighten(95%),
      stroke: 0.55pt + blue.lighten(55%),
      radius: 4pt,
      inset: 6pt,
      [#align(center + horizon, art)],
    ),
  ),
  caption: [*#title.* #note],
)

#let hero = cetz.canvas(length: 0.95cm, {
  import cetz.draw: *
  rect((0, 0), (13.7, 4.3), radius: .2, fill: blue.lighten(95%), stroke: .8pt + blue.lighten(55%))
  content((2.3, 3.8), text(weight: "bold", fill: navy)[tensor observations])
  for x in (0.4, .75, 1.05, 1.35, 1.7, 2.0, 2.25, 2.55) {
    circle((x, .75 + .8 * calc.sin(x * 1.8)), radius: .085, fill: blue, stroke: none)
  }
  for x in (2.8, 3.05, 3.2) { circle((x, 2.8), radius: .11, fill: red, stroke: none) }
  line((3.55, 2.0), (4.65, 2.0), mark: (end: ">"), stroke: 1.2pt + navy)
  rect((4.9, .45), (8.4, 3.55), radius: .15, fill: white, stroke: .8pt + gridline)
  content((6.65, 3.2), text(weight: "bold", fill: navy)[400-byte state])
  for i in range(0, 12) {
    let x = 5.15 + i * .25
    let y = .8 + 1.6 * calc.sin((i + 2) * .55) * calc.sin((i + 2) * .55)
    circle((x, y), radius: .07 + .018 * calc.rem(i, 3), fill: if i < 2 or i > 9 { orange } else { blue }, stroke: none)
  }
  line((8.65, 2.0), (9.75, 2.0), mark: (end: ">"), stroke: 1.2pt + navy)
  rect((10.0, .45), (13.3, 3.55), radius: .15, fill: white, stroke: .8pt + gridline)
  content((11.65, 3.2), text(weight: "bold", fill: navy)[later decisions])
  line((10.35, 1.1), (12.95, 1.1), stroke: 1pt + navy)
  for i in range(0, 9) { line((10.35 + i * .325, .95), (10.35 + i * .325, 1.3), stroke: .65pt + blue) }
  line((10.7, .7), (10.7, 2.65), stroke: 1.2pt + red)
  line((12.6, .7), (12.6, 2.65), stroke: 1.2pt + red)
  content((11.65, 2.82), text(size: 7.2pt, fill: muted)[quantiles · groups · ranges])
})

#let ptq-flow = cetz.canvas(length: 0.92cm, {
  import cetz.draw: *
  let boxes = ((.2, "tensors"), (3.1, "observations"), (6.0, "update"), (8.9, "RANKSTORE"), (11.8, "cold analysis"))
  for item in boxes {
    let x = item.at(0)
    rect((x, .5), (x + 2.35, 1.7), radius: .12, fill: if x == 8.9 { orange.lighten(88%) } else { blue.lighten(91%) }, stroke: .8pt + if x == 8.9 { orange } else { blue })
    content((x + 1.175, 1.1), text(size: 7.2pt, weight: "bold")[#item.at(1)])
  }
  for x in (2.55, 5.45, 8.35, 11.25) { line((x, 1.1), (x + .45, 1.1), mark: (end: ">"), stroke: 1pt + navy) }
  content((7.15, .15), text(size: 7.5pt, fill: muted)[observe now · choose policies later])
})

#let quantizer-art = cetz.canvas(length: 0.68cm, {
  import cetz.draw: *
  line((.4, 1.0), (9.4, 1.0), stroke: 1pt + navy)
  rect((.4, .65), (1.7, 1.35), fill: red.lighten(82%), stroke: none)
  rect((8.1, .65), (9.4, 1.35), fill: red.lighten(82%), stroke: none)
  for i in range(0, 9) {
    let x = 1.7 + i * .8
    line((x, .55), (x, 1.45), stroke: .8pt + blue)
  }
  line((1.7, .35), (1.7, 2.15), stroke: 1.3pt + red)
  line((8.1, .35), (8.1, 2.15), stroke: 1.3pt + red)
  circle((.8, 1.75), radius: .09, fill: orange, stroke: none)
  line((.8, 1.6), (1.7, 1.2), mark: (end: ">"), stroke: .9pt + orange)
  circle((5.25, 1.75), radius: .09, fill: green, stroke: none)
  line((5.25, 1.6), (5.3, 1.2), mark: (end: ">"), stroke: .9pt + green)
  content((1.7, 2.35), text(size: 7.5pt, fill: red)[lower clip])
  content((8.1, 2.35), text(size: 7.5pt, fill: red)[upper clip])
  content((5.0, .15), text(size: 7.5pt, fill: muted)[uniform integer bins])
})

#let loss-art = cetz.canvas(length: 0.68cm, {
  import cetz.draw: *
  line((.5, .35), (.5, 3.2), (9.3, 3.2), stroke: .8pt + navy)
  line((.7, .5), (1.1, .8), (1.5, 1.35), (2.0, 2.0), (2.7, 2.6), (3.8, 2.95), (5.0, 3.05), (6.2, 2.95), (7.3, 2.6), (8.0, 2.0), (8.5, 1.3), (9.0, .6), stroke: 1.2pt + blue)
  rect((.5, .35), (2.0, 3.2), fill: red.lighten(86%), stroke: none)
  rect((8.0, .35), (9.3, 3.2), fill: red.lighten(86%), stroke: none)
  for i in range(0, 7) {
    rect((2.0 + i * .86, .38), (2.65 + i * .86, .52 + .12 * calc.rem(i, 2)), fill: orange.lighten(45%), stroke: none)
  }
  line((2.0, .25), (2.0, 3.1), stroke: 1pt + red)
  line((8.0, .25), (8.0, 3.1), stroke: 1pt + red)
  content((1.15, 2.8), text(size: 7.5pt, fill: red)[clipped])
  content((5.0, 1.0), text(size: 7.5pt, fill: orange)[rounded inside])
  content((8.65, 2.8), text(size: 7.5pt, fill: red)[clipped])
})

#let measure-art = cetz.canvas(length: 0.68cm, {
  import cetz.draw: *
  content((2.1, 3.15), text(size: 7.5pt, weight: "bold", fill: navy)[thousands of samples])
  for i in range(0, 28) {
    let x = .35 + i * .14
    let y = .45 + .23 * calc.rem(i * 7, 9)
    circle((x, y), radius: .045, fill: blue.lighten(20%), stroke: none)
  }
  line((4.45, 1.55), (5.45, 1.55), mark: (end: ">"), stroke: 1.2pt + navy)
  content((7.65, 3.15), text(size: 7.5pt, weight: "bold", fill: navy)[64 weighted knots])
  for item in ((5.9, .8, .11), (6.5, 1.1, .18), (7.2, 1.55, .26), (8.0, 1.8, .30), (8.8, 1.25, .20), (9.4, .75, .12)) {
    circle((item.at(0), item.at(1)), radius: item.at(2), fill: orange.lighten(15%), stroke: .6pt + orange)
    line((item.at(0), .3), (item.at(0), item.at(1) - item.at(2)), stroke: .5pt + orange.lighten(40%))
  }
  content((7.7, .05), text(size: 7pt, fill: muted)[circle area = retained probability mass])
})

#let memory-art = cetz.canvas(length: 0.68cm, {
  import cetz.draw: *
  rect((.4, .8), (6.15, 2.1), fill: blue.lighten(82%), stroke: .8pt + blue)
  rect((6.15, .8), (9.05, 2.1), fill: orange.lighten(78%), stroke: .8pt + orange)
  rect((9.05, .8), (9.6, 2.1), fill: green.lighten(76%), stroke: .8pt + green)
  rect((9.6, .8), (10.15, 2.1), fill: purple.lighten(78%), stroke: .8pt + purple)
  content((3.25, 1.45), text(size: 8pt, weight: "bold")[64 values · 256 B])
  content((7.6, 1.45), text(size: 8pt, weight: "bold")[64 masses · 128 B])
  content((9.325, 2.55), text(size: 7pt, fill: green)[tie bits])
  content((9.875, .35), text(size: 7pt, fill: purple)[min/max])
  line((9.325, 2.05), (9.325, 2.35), stroke: .7pt + green)
  line((9.875, .55), (9.875, .8), stroke: .7pt + purple)
})

#let rank-art = cetz.canvas(length: 0.68cm, {
  import cetz.draw: *
  content((.45, 2.9), text(size: 7.5pt, weight: "bold")[uniform slot index])
  line((.5, 2.45), (9.3, 2.45), stroke: .8pt + navy)
  for i in range(0, 9) { line((.5 + i * 1.1, 2.28), (.5 + i * 1.1, 2.62), stroke: .7pt + blue) }
  content((.45, .15), text(size: 7.5pt, weight: "bold")[probability rank])
  line((.5, .65), (9.3, .65), stroke: .8pt + navy)
  for x in (.5, .62, .86, 1.25, 1.9, 2.9, 4.9, 6.9, 7.9, 8.55, 8.94, 9.18, 9.3) {
    line((x, .48), (x, .82), stroke: .7pt + orange)
  }
  for i in range(0, 9) {
    let top = .5 + i * 1.1
    let bottom = .5 + 8.8 * calc.sin(calc.pi * i / 16) * calc.sin(calc.pi * i / 16)
    line((top, 2.25), (bottom, .85), stroke: .45pt + gridline)
  }
  content((4.9, 1.45), text(size: 7.5pt, fill: muted)[more resolution near both tails])
})

#let mean-art = cetz.canvas(length: 0.68cm, {
  import cetz.draw: *
  line((.5, .65), (9.4, .65), stroke: .8pt + navy)
  for item in ((1.0, 1), (1.7, 1), (2.1, 2), (3.0, 1), (6.0, 2), (7.2, 1), (8.4, 1)) {
    let x = item.at(0)
    let w = item.at(1)
    circle((x, 1.15), radius: .09 + .045 * w, fill: blue.lighten(25%), stroke: none)
    line((x, .65), (x, 1.0), stroke: .5pt + blue)
  }
  line((4.55, .35), (4.55, 2.35), stroke: 1.4pt + orange)
  content((4.55, 2.65), text(size: 8pt, weight: "bold", fill: orange)[weighted balance point])
  line((1.0, 2.0), (4.4, 2.0), mark: (end: ">"), stroke: .8pt + blue)
  line((8.4, 2.0), (4.7, 2.0), mark: (end: ">"), stroke: .8pt + blue)
})

#let mass-art = cetz.canvas(length: 0.68cm, {
  import cetz.draw: *
  line((.6, .45), (.6, 3.2), (9.4, 3.2), stroke: .8pt + navy)
  line((.6, .45), (1.5, .72), (2.4, .96), (3.2, 1.25), (4.4, 1.55), (5.2, 1.95), (6.6, 2.25), (7.5, 2.7), (9.3, 3.15), stroke: 1.2pt + blue)
  line((.6, .45), (1.5, .45), (1.5, .72), (2.4, .72), (2.4, .98), (3.2, .98), (3.2, 1.24), (4.4, 1.24), (4.4, 1.56), (5.2, 1.56), (5.2, 1.94), (6.6, 1.94), (6.6, 2.26), (7.5, 2.26), (7.5, 2.71), (9.3, 2.71), (9.3, 3.15), stroke: 1pt + orange)
  content((3.1, 2.75), text(size: 7.5pt, fill: blue)[exact cumulative mass])
  content((6.7, 1.45), text(size: 7.5pt, fill: orange)[16-bit prefix rounding])
  line((8.15, 2.72), (8.15, 2.95), stroke: .7pt + red)
  content((8.2, 2.82), text(size: 7pt, fill: red)[tiny rank offset])
})

#let query-art = cetz.canvas(length: 0.68cm, {
  import cetz.draw: *
  line((.6, .45), (.6, 3.2), (9.4, 3.2), stroke: .8pt + navy)
  line((.7, .55), (1.7, .8), (2.7, 1.05), (3.5, 1.3), stroke: 1.25pt + blue)
  line((3.5, 1.3), (5.6, 1.3), stroke: 2pt + green)
  line((5.6, 1.3), (6.3, 2.05), (7.3, 2.35), (9.2, 3.05), stroke: 1.25pt + blue)
  for item in ((.7,.55),(1.7,.8),(2.7,1.05),(3.5,1.3),(5.6,1.3),(6.3,2.05),(7.3,2.35),(9.2,3.05)) { circle(item, radius: .07, fill: orange, stroke: none) }
  content((4.55, .85), text(size: 7.5pt, fill: green)[retained exact tie])
  content((2.0, 2.2), text(size: 7.5pt, fill: blue)[linear between mixed knots])
  content((9.15, .2), text(size: 7pt, fill: muted)[rank])
  content((.2, 3.05), text(size: 7pt, fill: muted)[value])
})

#let percentile-art = cetz.canvas(length: 0.68cm, {
  import cetz.draw: *
  line((.5, .4), (.5, 3.15), (9.3, 3.15), stroke: .8pt + navy)
  line((.7, .45), (1.2, .6), (1.8, 1.0), (2.5, 1.75), (3.4, 2.55), (4.4, 2.95), (5.4, 2.7), (6.3, 2.15), (7.1, 1.55), (7.8, .95), (8.5, .6), (9.1, .45), stroke: 1.2pt + blue)
  rect((.5, .4), (1.75, 3.15), fill: red.lighten(87%), stroke: none)
  rect((8.2, .4), (9.3, 3.15), fill: red.lighten(87%), stroke: none)
  line((1.75, .3), (1.75, 3.05), stroke: 1.2pt + red)
  line((8.2, .3), (8.2, 3.05), stroke: 1.2pt + red)
  content((1.1, 2.55), text(size: 7pt, fill: red)[discarded tail])
  content((8.7, 2.55), text(size: 7pt, fill: red)[discarded tail])
  content((5.0, .8), text(size: 7.5pt, fill: green)[kept dynamic range])
})

#let search-art = cetz.canvas(length: 0.68cm, {
  import cetz.draw: *
  content((2.0, 3.25), text(size: 7.5pt, weight: "bold")[candidate clipping windows])
  for item in ((.6, 1.5, red), (1.0, 2.8, orange), (1.4, 4.1, green)) {
    let y = item.at(0)
    let width = item.at(1)
    let tone = item.at(2)
    line((.5, y), (4.8, y), stroke: .5pt + gridline)
    line((2.65 - width/2, y - .18), (2.65 - width/2, y + .18), stroke: 1pt + tone)
    line((2.65 + width/2, y - .18), (2.65 + width/2, y + .18), stroke: 1pt + tone)
  }
  line((5.6, .45), (5.6, 3.1), (9.5, 3.1), stroke: .8pt + navy)
  line((5.9, 2.7), (6.5, 1.7), (7.1, .8), (7.6, .55), (8.2, .9), (8.8, 1.75), (9.3, 2.8), stroke: 1.3pt + blue)
  circle((7.6, .55), radius: .09, fill: green, stroke: none)
  content((7.6, .2), text(size: 7.5pt, fill: green)[best trade-off])
})

#let merge-art = cetz.canvas(length: 0.68cm, {
  import cetz.draw: *
  for row in range(0, 3) {
    rect((.4, .35 + row * 1.0), (3.7, 1.05 + row * 1.0), radius: .1, fill: blue.lighten(92%), stroke: .6pt + blue)
    for i in range(0, 6) {
      circle((.75 + i * .5, .7 + row * 1.0), radius: .055 + .012 * calc.rem(i + row, 3), fill: blue, stroke: none)
    }
  }
  line((4.05, 1.7), (5.2, 1.7), mark: (end: ">"), stroke: 1.2pt + navy)
  rect((5.5, .55), (9.6, 2.85), radius: .12, fill: orange.lighten(91%), stroke: .8pt + orange)
  content((7.55, 2.5), text(size: 8pt, weight: "bold")[one group distribution])
  for i in range(0, 9) {
    circle((5.9 + i * .4, .9 + .75 * calc.sin((i + 1) * .5) * calc.sin((i + 1) * .5)), radius: .06 + .015 * calc.rem(i, 4), fill: orange, stroke: none)
  }
  content((7.55, .25), text(size: 7pt, fill: muted)[union, sort, recompress once])
})

#let tie-art = cetz.canvas(length: 0.68cm, {
  import cetz.draw: *
  line((.6, .45), (.6, 3.2), (9.3, 3.2), stroke: .8pt + navy)
  line((.7, .65), (3.8, 1.15), stroke: 1.2pt + blue)
  line((3.8, 1.15), (3.8, 2.45), stroke: 2pt + green)
  line((3.8, 2.45), (9.1, 3.0), stroke: 1.2pt + blue)
  line((.7, 1.8), (9.1, 1.8), stroke: .9pt + orange, dash: "dashed")
  circle((3.8, 1.8), radius: .1, fill: orange, stroke: none)
  content((5.3, 1.55), text(size: 7.5pt, fill: orange)[requested rank lies inside jump])
  content((3.0, 2.75), text(size: 7.5pt, fill: green)[all these ranks correctly return one value])
})

#let bars-art = cetz.canvas(length: 1.05cm, {
  import cetz.draw: *
  line((1.9, .35), (1.9, 4.0), (9.6, 4.0), stroke: .7pt + navy)
  let rows = ((3.45, "uniform", .27, 1.41), (2.65, "normal", .27, 1.02), (1.85, "lognormal", .58, 4.45), (1.05, "bimodal", .93, 9.60))
  for row in rows {
    let y = row.at(0)
    let label = row.at(1)
    let a = row.at(2)
    let b = row.at(3)
    content((.9, y), text(size: 7.5pt)[#label])
    rect((1.95, y + .08), (1.95 + a * .65, y + .34), fill: orange, stroke: none)
    rect((1.95, y - .25), (1.95 + b * .65, y + .01), fill: blue, stroke: none)
  }
  content((7.4, .45), text(size: 7pt, fill: orange)[RANKSTORE])
  content((8.7, .45), text(size: 7pt, fill: blue)[t-digest])
  content((5.7, 3.75), text(size: 7pt, fill: muted)[mean rank error × 10³])
})

#align(center)[
  #v(0.20in)
  #text(size: 33pt, weight: "bold", fill: navy)[RANKSTORE]
  #v(0.10in)
  #text(size: 17pt, weight: "bold")[A 400-Byte Persistent Rank Summary]
  #text(size: 17pt, weight: "bold")[for Deferred Tensor Analysis]
  #v(0.15in)
  #text(size: 11.5pt, fill: muted)[Observe every sample now; choose quantiles, groups, diagnostics, and quantization later]
  #v(0.22in)
  #pill[tensor histories] #h(5pt)
  #pill([rank transport], tone: orange) #h(5pt)
  #pill([exact ties], tone: green) #h(5pt)
  #pill([cold decisions], tone: purple)
  #v(0.24in)
  #text(size: 9.5pt)[monatq research note · version 0.3]
]

#v(0.25in)
#hero
#v(0.18in)
#callout([The proposal], [Replace the current per-position t-digest with one deterministic weighted approximation of the empirical tensor history. Sixty-four value knots, 16-bit probability masses, one tie mask, and exact extrema occupy 400 bytes of summary state per position. The live collector also needs an input buffer and bounded flush workspace. Collection commits to no percentile, grouping, visualization, or quantizer; those decisions remain cold and revisable.], tone: orange)

#v(0.12in)
#callout([An important wording distinction], [RANKSTORE *ingests every observation*, but it does not retain every raw value. It preserves an approximate empirical distribution from which later rank-based decisions can be made. Raw samples cannot be reconstructed or replayed.], tone: red)

#pagebreak()
#outline(title: [Contents], depth: 2, indent: auto)

= Executive summary

`TensorDigest` receives a sequence of full tensor samples and tracks one distribution at every flat tensor position. The expensive part is not an individual value; it is the product of positions, samples, and persistent bytes. At collection time, callers often do not yet know which summaries they will need. They may later ask for robust ranges, percentiles, per-channel groups, visualizations, outlier diagnostics, or post-training quantization (PTQ) thresholds.

RANKSTORE separates *observation* from *decision*. During collection it maintains a compact positive approximation of each position’s empirical distribution. During a cold later stage it can answer quantiles, combine arbitrary positions into groups, evaluate range policies, or serve as input to a quantizer. Query and aggregation latency are explicitly secondary; update throughput and memory dominate.

#figure(ptq-flow, caption: [The intended scope. Every tensor observation updates the same compact state. Later consumers can choose their own grouping and decision policy without replaying the tensor stream.])

The proposed summary state uses 400 bytes per tensor position—roughly one twelfth of the repository t-digest’s 4,900-byte summary state at the default compression. It is 32 bytes, or 8.7%, larger than Quantile Spine’s 368-byte summary state. It contains no dynamic histogram, separately sized atom table, per-position heap allocation, or query-time cache. Equal values share the ordinary knot budget, and exact finite extrema remain available even though interior observations are compressed.

#callout([Memory scope], [The 400-byte figure is summary state, not complete live memory. With `f32` inputs, the proposed 256-row buffer adds 1,024 bytes per position. The previous comparison of 400 bytes against the t-digest’s complete 5.7 KiB allocation mixed those scopes and overstated the reduction.], tone: red)

#table(
  columns: (1.55fr, 1fr, 1fr, 1fr),
  align: (left, center, center, center),
  table.header([*Property*], [*t-digest*], [*Quantile Spine*], [*RANKSTORE*]),
  [Summary state/position], [4,900 B], [368 B], [400 B],
  [`f32` input buffer/position], [800 B], [1,024 B], [1,024 B],
  [Full sort buffer/position], [none], [1,024 B], [none proposed],
  [Tensor-scaled live total], [≈ 5,700 B], [2,416 B], [1,424 B],
  [State primitive], [weighted centroids], [rank anchors], [weighted rank knots],
  [Repeated values], [interpolated], [anchor dependent], [retained tie interval],
  [Exact min/max], [implementation dependent], [no], [yes, by construction],
  [Update], [buffer + centroid merge], [buffer + anchor blend], [buffer + fixed-array rebin],
  [Cold group merge], [centroid merge], [not available], [union + same rebin],
)

The table assumes default configurations, `f32`, a 64-bit target, and large tensors over which object headers and worker-local scratch amortize. The implemented t-digest and Quantile Spine live totals are verified from allocator-instrumented `backend_accuracy` runs rather than field-size estimates; the unimplemented RANKSTORE total remains a design projection. RANKSTORE’s 1,424-byte live estimate requires gathering and sorting into per-worker scratch rather than materializing a second tensor-sized buffer. Adding such a buffer would raise it to 2,448 bytes, slightly above the current Quantile Spine total. Conversely, Quantile Spine could adopt the same worker-local strategy and reach about 1,392 bytes. The live-memory difference between those two designs is therefore an implementation choice, not an inherent RANKSTORE advantage.

#callout([Evidence status], [Held-out Python experiments show better aggregate rank accuracy than the repository t-digest on smooth, atomic, quantized, and bimodal workloads. The 16-bit state transition and `f32` representatives are simulated. Rust throughput, exact live allocation, million-row drift, and downstream application quality remain unmeasured.], tone: green)

= Observe now, decide later

== Why fixed summaries are often too early

A minimum and maximum preserve endpoints but reveal nothing about how probability is distributed between them. A fixed histogram commits to bin edges before the eventual query is known. A single percentile commits to one tail policy. A preselected channel grouping prevents later comparison with per-tensor or hardware-aligned alternatives.

RANKSTORE instead records a reusable approximation of rank mass. The same collected state can support several later consumers:

- median, percentile bands, and robust ranges;
- per-position, per-channel, per-token, or whole-tensor aggregation;
- distribution plots and saturation diagnostics;
- anomaly or alert thresholds chosen after collection;
- PTQ clipping and scale selection;
- comparisons between alternative policies without replaying the model or data source.

The architecture is therefore closer to a compact tensor history than to a quantizer. PTQ is an important consumer, not the identity of the backend.

== What cannot be deferred

No fixed-memory summary retains raw sample identity, temporal order, correlations between positions, or exact arbitrary quantiles. If a later analysis needs sample-level replay, covariance, or causality, it must store different information. Comparison-based quantile sketches necessarily trade memory for error @karnin2016. RANKSTORE targets representative tensor distributions rather than a universal worst-case guarantee.

= A compact empirical distribution

== Positive weighted support

An observed stream defines an empirical distribution made of point masses. RANKSTORE replaces that large measure with at most 64 positive weighted locations. In one dimension, this can be understood as moving nearby rank mass onto representative support points—a small transport coreset @peyre2019.

#intuition(
  [The compressed history],
  [$ hat(mu)=sum_(j=1)^K p_j delta_(v_j), quad p_j>=0, quad sum_(j=1)^K p_j=1 $],
  measure-art,
  [Thousands of observations become a short weighted support. Large circles carry more probability; positivity and normalization are explicit.],
)

Every input affects the state, but most inputs cease to be individually identifiable after compression. Any number of distinct values may enter. At most 64 locations remain exact; excess support is approximated by the same rule used for continuous data. There is no separate “number of atoms supported” configuration.

== The 400-byte summary-state layout

All positions receive one observation at the same logical tensor update. With NaNs rejected for the whole update, the accepted sample count can live once at `TensorDigest` level. The per-position summary state spends its final eight bytes on exact extrema instead. Input buffering and flush workspace are separate from this layout.

#intuition(
  [Summary-state accounting],
  [$ 64 dot 4 + 64 dot 2 + 8 + 8 = 400 " bytes" $],
  memory-art,
  [The four terms are `f32` locations, `u16` masses, a 64-bit tie mask, and two `f32` extrema. Zero mass identifies an unused slot.],
)

#block(breakable: false)[
  #table(
    columns: (1.5fr, .7fr, 2.2fr),
    align: (left, right, left),
    table.header([*Field*], [*Bytes*], [*Meaning*]),
    [`values[64]: f32`], [256], [mixed-group means or retained exact values],
    [`masses[64]: u16`], [128], [probability quanta; active masses sum to 65,535],
    [`pure_mask: u64`], [8], [one bit per location that still represents one exact value],
    [`min`, `max`: `f32`], [8], [exact finite endpoints for later range decisions],
    [*Summary-state total*], [*400*], [no pointers and no per-position allocation],
  )
]

Per-position missingness would break the shared-count assumption and must pay an explicit memory or semantic cost. The initial backend should reject a whole tensor update containing NaN rather than silently desynchronize position counts.

= Streaming compression

== Buffer first, compress rarely

Updates append full row-major tensor samples to one 256-row input buffer. A flush gathers one position into worker-local scratch, then handles positions independently and in parallel:

+ Sort and run-length encode the new values.
+ Expand old normalized masses relative to the shared historical count.
+ Merge at most 64 old locations with the batch runs.
+ Place 64 target cells in rank space and snap cuts to indivisible masses.
+ Store a weighted mean for every mixed cell and a purity bit for every one-value cell.
+ Prefix-round cumulative masses back to 16 bits and update exact extrema.

The hot path uses bounded arrays over roughly 320 sorted entries. This scratch scales with the worker count, not the tensor width. The proposed implementation must not retain a `numel × batch_rows` sorted copy: that extra tensor-sized buffer would add another 1,024 bytes per `f32` position. There is no per-position hash table, dynamic allocation, iterative optimization, or query reconstruction.

== Tail-companded rank cells

Uniform rank cells spend the same resolution at the median and at the extremes. Many later decisions—robust ranges, alarms, saturation estimates, and quantization—care about tails. RANKSTORE therefore places fixed slots uniformly in an abstract coordinate that bends toward probability zero and one.

#intuition(
  [Tail-aware rank placement],
  [$ q(s)=sin^2(pi s/2), quad 0<=s<=1 $],
  rank-art,
  [Equal slot steps on the upper line become denser at both ends of probability space. This is the same broad tail-allocation principle that benefits t-digest @dunning2019.],
)

A large exact tie may contain several desired cuts. Those cuts snap to the tie boundaries; unused slots then recursively split the widest remaining intervals in transformed rank space. A 50% zero mass consumes one support location rather than half the state.

== Mixed cells use weighted means

A cell containing several locations stores their weighted mean. It is the constant representative minimizing squared value reconstruction error, and masses plus first moments combine cleanly before the next lossy repartition.

#intuition(
  [One representative per mixed cell],
  [$ v_G=(sum_(i in G) w_i x_i)/(sum_(i in G) w_i) $],
  mean-art,
  [The orange line is the weighted balance point. In the feasibility prototype, medians produced materially larger smooth-distribution error under cold interpolation.],
)

A cell containing one exact location retains that location and sets its purity bit. A mixed cell remains mixed even if its rounded mean happens to equal an input value, preventing accidental invention of a tie.

== Prefix-rounded probability

Sixteen-bit probability masses allow 64 locations at the same memory where 32-bit counts allow 48. The exact total sample count remains outside the cell. Cumulative prefixes are rounded first, then differenced, so every represented CDF boundary is directly controlled.

#intuition(
  [Sixteen-bit mass encoding],
  [$ C'_j="round"(65535 C_j/C_K), quad m_j=C'_j-C'_(j-1) $],
  mass-art,
  [Each orange step follows the blue cumulative mass. Prefix rounding avoids independent mass errors all drifting in one direction.],
)

The current boundary discrepancy is below one half probability quantum. Repeated compression and requantization can still accumulate approximation; long streams and repeated distribution shifts remain mandatory tests.

= Cold reconstruction and regrouping

== A monotone quantile curve

Each mixed knot is anchored at the center rank of its encoded mass. Neighboring mixed anchors are joined linearly. A pure knot contributes a horizontal interval across its mass. Sorting values and ranks makes the resulting quantile curve monotone.

#intuition(
  [Decode only when needed],
  [$ r_j=(C_(j-1)+C_j)/(2C_K), quad hat(Q)(r_j)=v_j $],
  query-art,
  [Blue segments interpolate unresolved continuous mass. The green plateau represents an exact retained tie across a whole rank interval.],
)

Endpoint requests bypass this interpolation. Probability zero returns exact `min`; probability one returns exact `max`. This avoids the severe heavy-tail underestimation that would result from treating the outer mixed-group means as extrema.

== Choose robust ranges later

A caller can defer its tail policy until after collection. One consumer may want a central visualization band; another may choose an anomaly threshold; another may estimate a clipping interval. All can use different tail probabilities from the same state.

#intuition(
  [A deferred central range],
  [$ a=Q(tau/2), quad b=Q(1-tau/2) $],
  percentile-art,
  [The shaded total tail mass is selected later. Collection does not hardcode `tau`, symmetry, or the eventual use of the range.],
)

== Merge arbitrary positions later

The stored object is a positive weighted measure. To form a channel, token group, spatial region, or whole tensor, scale cell masses by their observation counts and optional group weights, union all support, sort once, and run the same compressor.

#intuition(
  [Deferred grouping],
  [$ hat(mu)_G=sum_(c in G) n_c/(sum_(d in G) n_d) hat(mu)_c $],
  merge-art,
  [Fine-grained histories become one group distribution without replaying the original tensor samples. Grouping policy is a cold decision.],
)

Aggregation latency is intentionally not a performance requirement. Cold merges may allocate temporary vectors and globally sort all input support.

= Example consumer: post-training quantization

PTQ illustrates why deferred distribution storage is useful. A calibration run can collect activation histories before choosing bit width, symmetric versus asymmetric ranges, per-tensor versus per-channel grouping, or clipping policy. The same RANKSTORE state can compare those choices later. PTQ methods commonly use a small representative set to establish activation ranges @hubara2021 @nagel2021.

== Evaluate an affine quantizer

A bounded real interval determines the step spacing of a finite integer codebook @gray1998 @jacob2018. Wider endpoints preserve outliers but make every interior step coarser.

#intuition(
  [One possible deferred quantizer],
  [$ Delta = (b-a)/(2^B-1) $ #linebreak() $ hat(x)=a+Delta "round"(("clip"(x,a,b)-a)/Delta) $],
  quantizer-art,
  [The red endpoints and bit width are not chosen while observations are collected. They are candidate decisions evaluated afterward.],
)

== Search candidates on the stored measure

Instead of selecting one percentile in advance, a cold quantizer can evaluate candidate intervals directly on the 64 weighted locations. This approximates activation reconstruction error while making search cost independent of the original sample count.

#intuition(
  [Quantization is a consumer, not the state],
  [$ hat(L)(a,b)=sum_(j=1)^K p_j (v_j-hat(v_j)_(a,b))^2 $],
  search-art,
  [Narrow, medium, and wide candidate windows trade clipping against rounding. Other consumers can ignore this objective entirely.],
)

This marginal objective does not model Hessians, cross-channel correlations, or final task loss. More sophisticated quantization methods can consume the same distributional evidence alongside their own model-specific information.

= Accuracy measurement

== Ties require rank intervals

A requested quantile inside an empirical jump is correctly answered by the repeated value. Measuring only one side of the CDF would report an error even though the returned value is a valid generalized quantile @hyndman1996.

#intuition(
  [Tie-aware rank error],
  [$ e_r(q,x)=max(F^-(x)-q, q-F(x), 0) $],
  tie-art,
  [The orange requested rank lies inside the green CDF jump, so returning the repeated value has zero error.],
)

Application-specific consumers must report their own outcomes as well. Rank error evaluates the recorder; it does not replace downstream quality, reconstruction error, alert precision, or visualization usefulness.

== Feasibility protocol

The checked-in experiment uses two held-out seeds, 32,768 `f32` observations per workload, 256-row flushes, and 235 query probabilities from one ten-thousandth through its upper complement. Parameter exploration used a separate seed. The Python prototype simulates 16-bit prefix masses and persistent `f32` representatives. Repository `TDigest` and `QuantileSpine` run through a temporary Rust crate on identical ordered values.

Workloads include uniform, normal, two lognormal scales, Student-t, shuffled and blocked mixtures, one and five added atoms, 50% zero, constants, 8–256-level quantization, and identical uniform multisets in random, ascending, and descending order. Ordered cases are diagnostics rather than proposed gates.

== General-distribution results

#table(
  columns: (1.35fr, .9fr, .9fr, .9fr, .9fr),
  align: (left, right, right, right, right),
  table.header([*Workload*], [*RANKSTORE mean*], [*t-digest mean*], [*RANKSTORE max*], [*t-digest max*]),
  [uniform], [0.000267], [0.001414], [0.001150], [0.005179],
  [normal], [0.000271], [0.001018], [0.001069], [0.005746],
  [lognormal, σ=1], [0.000346], [0.002420], [0.001304], [0.008954],
  [lognormal, σ=2], [0.000584], [0.004453], [0.001637], [0.011169],
  [Student-t, df=2], [0.000392], [0.000861], [0.001288], [0.004347],
  [bimodal, shuffled], [0.000927], [0.009599], [0.013983], [0.066852],
  [bimodal, blocked], [0.001288], [0.010447], [0.016979], [0.066882],
)

#figure(bars-art, caption: [Mean valid-rank error on selected held-out workloads. Bars use thousandths of rank. Lower is better; separated modes remain harder than smooth distributions.])

Across all 36 workload/seed pairs, RANKSTORE’s mean-of-means was 0.000330 and its worst observed maximum was 0.016979. The repository t-digest values were 0.006928 and 0.246068. RANKSTORE had lower mean and maximum error on every nonconstant pair; both methods tied on constants.

== Discrete observations

#block(breakable: false)[
  #table(
    columns: (1.5fr, 1fr, 1fr, 1.2fr),
    align: (left, right, right, left),
    table.header([*Workload*], [*Mean error*], [*Worst error*], [*Interpretation*]),
    [five added atoms], [0.000117], [0.001653], [ties share the ordinary knot budget],
    [8 levels], [0], [0], [all levels retained],
    [16 levels], [0], [0], [all levels retained],
    [32 levels], [0], [0], [all levels retained],
    [64 levels], [0], [0], [all levels retained in this test],
    [256 levels], [0.000831], [0.002344], [excess levels interpolated],
  )
]

Zero means zero at the 235 tested ranks, not equality at every possible probability. The result demonstrates that repeated values need no separately hardcoded capacity.

== Important caveats

- *Extreme tails:* t-digest was often two to three times more accurate at individual smooth tail queries, although RANKSTORE’s absolute tail errors were small. The worst observed tail error was 0.001719.
- *Separated modes:* linear interpolation across a large gap remains the largest ordinary-distribution error.
- *Extrema:* the experiment’s interior state did not include endpoint requests. Proposed min/max sidecars make endpoints exact by construction without changing reported interior estimates.
- *No speed evidence:* Python runtime is not a proxy for Rust tensor throughput.
- *No consumer evidence:* no downstream quantizer, alert, or visualization study has yet compared decisions from this state.

= A deliberately small systems design

== Rust representation

```rust
#[repr(C)]
struct RankStore {
    values: [f32; 64],
    masses: [u16; 64],
    pure_mask: u64,
    min: f32,
    max: f32,
}
```

The accepted tensor sample count and exceptional-value policy belong to `TensorDigest`, shared across all positions. Temporary flush arithmetic uses wider accumulators; persistent representatives round to `f32` only once per flush. With the default 256 `f32` rows, live tensor-scaled storage is therefore 400 + 1,024 = 1,424 bytes per position, not 400 bytes. Per-worker scratch and small object headers remain additional but do not scale with the number of positions.

== One compressor everywhere

The same deterministic reducer should implement buffered flush and every cold merge. A single representation before and after aggregation avoids separate “streaming,” “query,” and “merge” models. Cold consumers may build caches, dense plots, or candidate-search workspaces, but none become persistent per-position state.

== Minimal configuration

#table(
  columns: (1.35fr, 1fr, 2.1fr),
  table.header([*Parameter*], [*Default*], [*Reason*]),
  [`knots`], [`64`], [fills the 400-byte packed layout],
  [`mass_quanta`], [`65535`], [full positive `u16` probability range],
  [`batch_rows`], [`256`], [matches the experiment; costs 1,024 input-buffer bytes per `f32` position],
  [`rank_scale`], [`arcsine`], [balanced measured accuracy with tail emphasis],
  [`NaN`], [`reject update`], [preserves one shared count across positions],
  [`infinities`], [`explicit endpoints`], [avoid contaminating finite means],
)

Do not expose the mass encoding or scale function publicly until long-stream and cross-platform tests stabilize the state transition.

= Validation plan

== Recorder-level gates

#table(
  columns: (1.2fr, 2.8fr),
  table.header([*Gate*], [*Requirement*]),
  [Typical accuracy], [lower aggregate mean and maximum rank error than t-digest on the preregistered representative suite],
  [Memory], [400 bytes of summary state and at most 1,424 tensor-scaled live bytes per `f32` position at 256 rows; no tensor-sized sorted copy],
  [Hot throughput], [update and flush throughput at least as high as the current t-digest at target tensor widths],
  [Numerics], [monotone queries, finite mixed means, exact finite extrema, and no mass underflow],
  [Long streams], [no unacceptable drift through at least one million updates and repeated distribution shifts],
  [Cold merge], [grouped summaries agree with direct-stream summaries within a declared rank tolerance],
)

Ascending streams, blocked modes, and crafted float patterns remain non-gating diagnostics. They still need a loose catastrophic ceiling: an unlikely stream is not permission for NaNs or nonmonotone answers.

== Consumer-level studies

Recorder accuracy is necessary but not sufficient. Validate at least one real deferred consumer from each intended class:

- a visualization or percentile-band report;
- a grouped channel/tensor summary;
- a PTQ calibration comparison against t-digest-selected ranges;
- a thresholding or saturation diagnostic if those APIs are planned.

The consumer study should use the same collected state to compare several policies. That is the central benefit of deferral.

= Risks and limitations

- *Approximation, not archival:* individual observations and temporal order are irrecoverable.
- *Representative data:* no summary can infer a deployment regime that was never observed.
- *Marginal state:* cross-position covariance and sample identity are absent.
- *Requantization drift:* 16-bit masses buy 64 support locations but repeatedly approximate old probability. A 48-location `u32` fallback remains useful.
- *Shared count:* rejecting an entire update because one position is NaN may not suit every caller. Per-position missingness requires a revised layout.
- *Tail parity:* t-digest remains stronger at many individual extreme-tail queries.
- *Working memory:* the 400-byte title names summary state only. The default input buffer raises proposed live tensor-scaled storage to 1,424 bytes per `f32` position, and a full sorted copy would raise it to 2,448 bytes.
- *Fixed memory:* no 400-byte summary state offers tiny distribution-free error for every stream.

#callout([The design principle], [Collection should preserve a useful approximation of the observed distribution while making as few downstream choices as possible. RANKSTORE spends its state budget on rank mass, ties, and extrema; everything else is deferred.], tone: green)

= Reproducibility

The executable experiment and held-out results are:

- `docs/experiments/transport_coreset_feasibility.py`
- `docs/experiments/results/transport-coreset-holdout.csv`

The script deterministically generates all workloads, simulates both 48-location `u32` and 64-location `u16` streaming layouts, creates a temporary Rust crate, and queries repository `TDigest` and `QuantileSpine`. It requires Python, NumPy, and Cargo. The offline row is a representation upper bound, not a streaming backend.

The next artifact should record real tensor shapes, sample counts, throughput, selected cold policies, and consumer outputs in a machine-readable table.

= Conclusion

RANKSTORE is a compact tensor-history backend, not a quantizer. It ingests every observation into a 400-byte positive summary state, preserves retained ties and exact extrema, and allows quantiles, grouping, visualization, clipping, and quantization policies to be chosen later. In the current feasibility suite it improves broad rank accuracy and discrete behavior over the repository t-digest. Apples-to-apples, its summary state is about one twelfth of the default t-digest state; its proposed one-buffer live tensor-scaled allocation is about one quarter of the current t-digest allocation. Its summary state is 32 bytes larger than Quantile Spine’s, so any live-memory advantage over that backend depends on avoiding Quantile Spine’s tensor-sized sort buffer.

The remaining questions are operational: Rust throughput and exact live allocation, long-stream mass drift, cold-merge fidelity, and whether real deferred consumers make decisions as good as or better than those based on t-digest. Those tests should decide the backend before its public API is stabilized.

#bibliography("references.bib", title: [References])
