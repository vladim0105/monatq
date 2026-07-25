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
      RankKnot · compact tensor histories for deferred analysis
      #h(1fr)
      Research note · version 0.4
    ]
  },
  footer: context {
    if counter(page).get().first() > 1 [
      #set text(size: 8pt, fill: muted)
      Current K32 Rust backend · historical K64 feasibility evidence
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
  content((6.65, 3.2), text(weight: "bold", fill: navy)[208-byte state])
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
  let boxes = ((.2, "tensors"), (3.1, "observations"), (6.0, "update"), (8.9, "RankKnot"), (11.8, "cold analysis"))
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
  content((7.65, 3.15), text(size: 7.5pt, weight: "bold", fill: navy)[up to 32 weighted knots])
  for item in ((5.9, .8, .11), (6.5, 1.1, .18), (7.2, 1.55, .26), (8.0, 1.8, .30), (8.8, 1.25, .20), (9.4, .75, .12)) {
    circle((item.at(0), item.at(1)), radius: item.at(2), fill: orange.lighten(15%), stroke: .6pt + orange)
    line((item.at(0), .3), (item.at(0), item.at(1) - item.at(2)), stroke: .5pt + orange.lighten(40%))
  }
  content((7.7, .05), text(size: 7pt, fill: muted)[circle area = retained probability mass])
})

#let memory-art = cetz.canvas(length: 0.68cm, {
  import cetz.draw: *
  rect((.4, .8), (6.4, 2.1), fill: blue.lighten(82%), stroke: .8pt + blue)
  rect((6.4, .8), (9.4, 2.1), fill: orange.lighten(78%), stroke: .8pt + orange)
  rect((9.4, .8), (9.775, 2.1), fill: green.lighten(76%), stroke: .8pt + green)
  rect((9.775, .8), (10.15, 2.1), fill: purple.lighten(78%), stroke: .8pt + purple)
  content((3.4, 1.45), text(size: 8pt, weight: "bold")[32 values · 128 B])
  content((7.9, 1.45), text(size: 8pt, weight: "bold")[32 masses · 64 B])
  content((9.5875, 2.55), text(size: 7pt, fill: green)[tie bits])
  content((9.9625, .35), text(size: 7pt, fill: purple)[min/max])
  line((9.5875, 2.05), (9.5875, 2.35), stroke: .7pt + green)
  line((9.9625, .55), (9.9625, .8), stroke: .7pt + purple)
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
  content((7.4, .45), text(size: 7pt, fill: orange)[K64 prototype])
  content((8.7, .45), text(size: 7pt, fill: blue)[t-digest])
  content((5.7, 3.75), text(size: 7pt, fill: muted)[mean rank error × 10³])
})

#align(center)[
  #v(0.20in)
  #text(size: 33pt, weight: "bold", fill: navy)[RankKnot]
  #v(0.10in)
  #text(size: 17pt, weight: "bold")[A 208-Byte In-Memory Rank Summary]
  #text(size: 17pt, weight: "bold")[for Deferred Tensor Analysis]
  #v(0.15in)
  #text(size: 11.5pt, fill: muted)[Observe every sample now; choose quantiles, groups, diagnostics, and quantization later]
  #v(0.22in)
  #pill[tensor histories] #h(5pt)
  #pill([rank transport], tone: orange) #h(5pt)
  #pill([exact ties], tone: green) #h(5pt)
  #pill([cold decisions], tone: purple)
  #v(0.24in)
  #text(size: 9.5pt)[monatq implementation note · version 0.4]
]

#v(0.25in)
#hero
#v(0.18in)
#callout([The current implementation], [`TensorDigest<f32, RankKnot>` is an optional statically selected kernel. Thirty-two value knots, 16-bit probability masses, one tie mask, and exact extrema occupy 208 bytes of state per position. The live collector also uses a 256-row input buffer by default and worker-local flush scratch. The generic API includes merge, analysis, zero-filtering, serialization, and optional visualization methods, but RankKnot’s implementations are explicit stubs that panic; direct support access and quantization remain future work.], tone: orange)

#v(0.12in)
#callout([An important wording distinction], [RankKnot incorporates every supported observation that reaches a successful flush, but it does not retain raw values afterward. It preserves an approximate empirical distribution from which later rank-based decisions can be made. Raw samples, their temporal order, and cross-position relationships cannot be reconstructed. NaN input is unsupported and currently fails during flush rather than being rejected atomically by `update`.], tone: red)

#pagebreak()
#outline(title: [Contents], depth: 2, indent: auto)

= Executive summary

`TensorDigest` receives full tensor samples and tracks one distribution at every flat tensor position. `TensorDigest<f32, RankKnot>` is now an implemented, statically selected backend optimized for update throughput and compact state. It buffers complete samples, compresses each position independently in parallel, and exposes per-position quantiles and exact extrema.

RankKnot separates *observation* from *decision*, but today it covers only recording and rank queries. Generic `TensorDigest` methods provide construction, shape, weight, update, flush, quantile, merge, analysis, zero filtering, serialization, and optional visualization operations; RankKnot adds sample count, configuration, and tensor/cell extrema. Its unsupported generic operations currently panic as unimplemented. Direct knot access, working grouping, persistence, visualization, analysis, and PTQ search remain design directions.

#figure(ptq-flow, caption: [The current recorder. Every supported tensor observation updates the same compact state. Quantiles and extrema are available now; richer cold consumers require future APIs.])

The K32 state uses 208 bytes per tensor position: 32 `f32` values, 32 `u16` masses, a 64-bit purity mask, and two `f32` extrema. This is about one twenty-fourth of the repository t-digest’s 4,900-byte summary state and 160 bytes smaller than Quantile Spine’s 368-byte state. Equal values share the ordinary knot budget, and extrema remain exact even though interior observations are compressed.

#callout([Memory scope], [The 208-byte figure is summary state, not complete live memory. With `f32` inputs, the default 256-row buffer adds 1,024 bytes per position, for approximately 1,232 tensor-scaled retained bytes per position. Worker-local vectors and object headers add a small non-tensor-scaled overhead and appear in peak allocator measurements.], tone: red)

#table(
  columns: (1.55fr, 1fr, 1fr, 1fr),
  align: (left, center, center, center),
  table.header([*Property*], [*t-digest*], [*Quantile Spine*], [*RankKnot K32*]),
  [Summary state/position], [4,900 B], [368 B], [208 B],
  [`f32` input buffer/position], [800 B], [1,024 B], [1,024 B],
  [Additional tensor sort buffer], [none], [1,024 B], [none],
  [Approx. tensor-scaled retained total], [5,700 B], [2,416 B], [1,232 B],
  [State primitive], [weighted centroids], [rank anchors], [up to 32 weighted knots],
  [Repeated values], [interpolated], [anchor dependent], [retained pure interval],
  [Exact min/max], [yes], [yes], [yes],
  [Public `merge_*` contract], [implemented], [stub: panics], [stub: panics],
)

The table assumes default configurations, `f32`, and a 64-bit target. Allocator instrumentation over 32 positions measured RankKnot at 39,432 live bytes and 45,064 peak bytes, versus 182,408 live bytes for t-digest and 77,320 for Quantile Spine. The component totals explain the scaling; exact allocator totals also include shape vectors, headers, and worker scratch.

#callout([Evidence status], [Current Rust K32 accuracy and memory are measured by `backend_accuracy`, and the repository benchmark exercises update throughput. A local Apple M4 run is reported below. The broader checked-in Python study remains historical K64 feasibility evidence, not a measurement of the current K32 backend. Million-row drift and downstream application quality remain open.], tone: green)

= Observe now, decide later

== Why fixed summaries are often too early

A minimum and maximum preserve endpoints but reveal nothing about how probability is distributed between them. A fixed histogram commits to bin edges before the eventual query is known. A single percentile commits to one tail policy. A preselected channel grouping prevents later comparison with per-tensor or hardware-aligned alternatives.

RankKnot instead records a reusable approximation of rank mass. The current API supports medians, arbitrary quantiles, percentile bands assembled by the caller, and exact per-position extrema. The encoded representation could also support grouping, distribution plots, saturation diagnostics, alert thresholds, and PTQ clipping if dedicated consumers or public state access are added.

#callout([API status], [One generic `TensorDigest` contract now includes merge, analysis, zero filtering, serialization, and optional visualization. T-digest implements those operations; RankKnot and Quantile Spine expose them through explicit `unimplemented!` stubs that panic. RankKnot state remains crate-private, and callers cannot enumerate knots or hand weighted support directly to a quantizer.], tone: purple)

The architecture is closer to a compact tensor history than to a quantizer, but today only its per-position rank-query surface is implemented.

== What cannot be deferred

No fixed-memory summary retains raw sample identity, temporal order, correlations between positions, or exact arbitrary quantiles. If a later analysis needs sample-level replay, covariance, or causality, it must store different information. Comparison-based quantile sketches necessarily trade memory for error @karnin2016. RankKnot targets representative tensor distributions rather than a universal worst-case guarantee.

= A compact empirical distribution

== Positive weighted support

An observed stream defines an empirical distribution made of point masses. RankKnot replaces that large measure with at most 32 positive weighted locations. In one dimension, this can be understood as moving nearby rank mass onto representative support points—a small transport coreset @peyre2019.

#intuition(
  [The compressed history],
  [$ hat(mu)=sum_(j=1)^K p_j delta_(v_j), quad p_j>=0, quad sum_(j=1)^K p_j=1 $],
  measure-art,
  [Thousands of observations become a short weighted support. Large circles carry more probability; positivity and normalization are explicit.],
)

Every input affects the state, but most inputs cease to be individually identifiable after compression. Any number of distinct values may enter. At most 32 locations remain exact; excess support is approximated by the same rule used for continuous data. There is no separate “number of atoms supported” configuration.

== The 208-byte K32 summary-state layout

All positions receive one observation at the same logical tensor update, so one `u64` sample count lives in RankKnot storage rather than in every position. The per-position state spends its final eight bytes on exact extrema. Input buffering and flush workspace are separate from this layout. NaN input is unsupported: `update` currently buffers it without validation, and a later flush panics in the sort comparator. This is not an atomic whole-update rejection contract.

#intuition(
  [Summary-state accounting],
  [$ 32 dot 4 + 32 dot 2 + 8 + 8 = 208 " bytes" $],
  memory-art,
  [The four terms are `f32` locations, `u16` masses, a 64-bit tie mask, and two `f32` extrema. Zero mass identifies an unused slot.],
)

#block(breakable: false)[
  #table(
    columns: (1.5fr, .7fr, 2.2fr),
    align: (left, right, left),
    table.header([*Field*], [*Bytes*], [*Meaning*]),
    [`values[32]: f32`], [128], [mixed-group means or retained exact values],
    [`masses[32]: u16`], [64], [probability quanta; active masses sum to 65,535],
    [`pure_mask: u64`], [8], [one bit per active location; upper 32 bits are unused],
    [`min`, `max`: `f32`], [8], [exact endpoints, including supported infinities],
    [*Summary-state total*], [*208*], [no pointers and no per-position allocation],
  )
]

Per-position missingness would break the shared-count assumption and must pay an explicit memory or semantic cost. Atomic NaN rejection would require an ingestion validation pass; the current performance-oriented contract instead assumes NaN-free observations. Positive and negative infinity are supported and protected as pure singleton groups.

= Streaming compression

== Buffer first, compress rarely

Updates append full row-major tensor samples to one input buffer whose public `buffer_capacity` defaults to 256. A flush handles positions independently using Rayon, with a minimum parallel chunk length of 64:

+ Gather one position into a worker-local `Vec<f32>` and unstable-sort it with `partial_cmp`.
+ Linearly merge it with up to 32 old representatives, coalescing equal values during the merge.
+ Evaluate 31 fixed arcsine cuts and snap each to the nearest entry boundary.
+ Skip duplicate or unusable cuts; fewer than 32 groups may remain. Protect infinities as pure singleton groups.
+ Store exact pure singletons and `f64`-accumulated weighted means, rounded to `f32`, for mixed groups.
+ Prefix-round cumulative masses to 65,535 with ties-to-even, difference adjacent prefixes, and update extrema.

At the default capacity the merged stream contains at most 288 entries. Scratch consists of two preallocated worker-local vectors and one fixed 31-element boundary array. It scales with parallel work rather than tensor width; there is no `numel × batch_rows` sorted copy, per-position hash table, or iterative optimizer. The vectors do allocate when a Rayon job initializes, so “no dynamic allocation” applies only to persistent per-position state, not flush workspace.

== Tail-companded rank cells

Uniform rank cells spend the same resolution at the median and at the extremes. Many later decisions—robust ranges, alarms, saturation estimates, and quantization—care about tails. RankKnot therefore places fixed slots uniformly in an abstract coordinate that bends toward probability zero and one.

#intuition(
  [Tail-aware rank placement],
  [$ q(s)=sin^2(pi s/2), quad 0<=s<=1 $],
  rank-art,
  [Equal slot steps on the upper line become denser at both ends of probability space. This is the same broad tail-allocation principle that benefits t-digest @dunning2019.],
)

A large exact tie may contain several desired cuts. Duplicate snapped cuts are skipped, so a 50% zero mass consumes one support location rather than half the state. The current implementation does not recursively refill unused cuts; a tie-heavy stream can therefore finish with fewer than 32 active locations.

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

Sixteen-bit probability masses keep the K32 state compact. The exact total sample count remains outside each position. Cumulative prefixes are rounded first, then differenced, so every represented CDF boundary is directly controlled.

#intuition(
  [Sixteen-bit mass encoding],
  [$ C'_j="round"(65535 C_j/C_K), quad m_j=C'_j-C'_(j-1) $],
  mass-art,
  [Each orange step follows the blue cumulative mass. Prefix rounding avoids independent mass errors all drifting in one direction.],
)

The current boundary discrepancy is at most one half probability quantum. Repeated compression and requantization can still accumulate approximation; long streams and repeated distribution shifts remain mandatory tests.

= Cold reconstruction and regrouping

== A monotone quantile curve

Each mixed knot is anchored at the center rank of its encoded mass. Neighboring mixed anchors are joined linearly. A pure knot contributes a horizontal interval across its mass. Sorting values and ranks makes the resulting quantile curve monotone.

#intuition(
  [Decode only when needed],
  [$ r_j=(C_(j-1)+C_j)/(2C_K), quad hat(Q)(r_j)=v_j $],
  query-art,
  [Blue segments interpolate unresolved continuous mass. The green plateau represents an exact retained tie across a whole rank interval.],
)

Endpoint requests bypass this interpolation. For a nonempty summary, any `q <= 0` returns exact `min`, any `q >= 1` returns exact `max`, and a NaN probability returns NaN. An empty summary returns `0.0` for every probability, including NaN, because the emptiness check runs first. Queries flush pending rows first and can therefore trigger the unsupported-input panic if buffered observations contain NaN. Tensor-wide queries run in parallel; `cell_quantiles` evaluates one position locally. When an infinity is adjacent to a finite anchor, the interpolation helper returns the infinity throughout that gap rather than producing NaN.

== Choose robust ranges later

A caller can defer its tail policy until after collection. One consumer may want a central visualization band; another may choose an anomaly threshold; another may estimate a clipping interval. All can use different tail probabilities from the same state.

#intuition(
  [A deferred central range],
  [$ a=Q(tau/2), quad b=Q(1-tau/2) $],
  percentile-art,
  [The shaded total tail mass is selected later. Collection does not hardcode `tau`, symmetry, or the eventual use of the range.],
)

== Future consumer: regrouping

The generic contract now exposes `merge_cells`, `merge_channels`, and `merge_all`, but RankKnot’s implementations are deliberate `unimplemented!` stubs and panic when called. Its encoded state is crate-private, so callers cannot provide the missing union externally. A working implementation could treat each stored object as a positive weighted measure: scale masses by observation counts and optional group weights, union support, sort once, and run a compatible compressor.

#intuition(
  [Deferred grouping],
  [$ hat(mu)_G=sum_(c in G) n_c/(sum_(d in G) n_d) hat(mu)_c $],
  merge-art,
  [Fine-grained histories become one group distribution without replaying the original tensor samples. Grouping policy is a cold decision.],
)

Aggregation latency would not be a performance requirement. Such a future cold merge could allocate temporary vectors and globally sort all input support. The present reducer is used only for buffered ingestion.

= Future consumer: post-training quantization

PTQ illustrates why deferred distribution storage could be useful. A calibration run could collect activation histories before choosing bit width, symmetric versus asymmetric ranges, per-tensor versus per-channel grouping, or clipping policy. No current RankKnot API exposes weighted knots to a quantizer, and no quantizer consumes this state; the following equations describe a proposed consumer. PTQ methods commonly use a small representative set to establish activation ranges @hubara2021 @nagel2021.

== Evaluate an affine quantizer

A bounded real interval determines the step spacing of a finite integer codebook @gray1998 @jacob2018. Wider endpoints preserve outliers but make every interior step coarser.

#intuition(
  [One possible deferred quantizer],
  [$ Delta = (b-a)/(2^B-1) $ #linebreak() $ hat(x)=a+Delta "round"(("clip"(x,a,b)-a)/Delta) $],
  quantizer-art,
  [The red endpoints and bit width are not chosen while observations are collected. They are candidate decisions evaluated afterward.],
)

== Search candidates on the stored measure

With a future support-export or dedicated PTQ API, a cold quantizer could evaluate candidate intervals directly on up to 32 weighted locations. This would approximate activation reconstruction error while making search cost independent of the original sample count.

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

== Historical K64 Python feasibility protocol

The checked-in historical experiment uses two held-out seeds, 32,768 `f32` observations per workload, 256-row K64 prototype flushes, and 235 query probabilities from one ten-thousandth through its upper complement. Parameter exploration used a separate seed. The Python prototype simulates a *64-knot* state with 16-bit prefix masses and persistent `f32` representatives. Repository `TDigest` and `QuantileSpine` run through a temporary Rust crate on identical ordered values using their own default buffering. These measurements do not describe the current K32 Rust implementation.

Workloads include uniform, normal, two lognormal scales, Student-t, shuffled and blocked mixtures, one and five added atoms, 50% zero, constants, 8–256-level quantization, and identical uniform multisets in random, ascending, and descending order. Ordered cases are diagnostics rather than proposed gates.

== Historical K64 general-distribution results

#table(
  columns: (1.35fr, .9fr, .9fr, .9fr, .9fr),
  align: (left, right, right, right, right),
  table.header([*Workload*], [*K64 mean*], [*t-digest mean*], [*K64 max*], [*t-digest max*]),
  [uniform], [0.000267], [0.001414], [0.001150], [0.005179],
  [normal], [0.000271], [0.001018], [0.001069], [0.005746],
  [lognormal, σ=1], [0.000346], [0.002420], [0.001304], [0.008954],
  [lognormal, σ=2], [0.000584], [0.004453], [0.001637], [0.011169],
  [Student-t, df=2], [0.000392], [0.000861], [0.001288], [0.004347],
  [bimodal, shuffled], [0.000927], [0.009599], [0.013983], [0.066852],
  [bimodal, blocked], [0.001288], [0.010447], [0.016979], [0.066882],
)

#figure(bars-art, caption: [Historical K64 Python mean valid-rank error on selected held-out workloads. Bars use thousandths of rank. Lower is better; separated modes remain harder than smooth distributions.])

Across all 36 workload/seed pairs, the historical K64 prototype’s mean-of-means was 0.000330 and its worst observed maximum was 0.016979. The repository t-digest values were 0.006928 and 0.246068. K64 had lower mean and maximum error on every nonconstant pair; both methods tied on constants. This is capacity-design evidence, not a claim about current K32.

== Historical K64 discrete observations

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

Zero means zero at the 235 tested ranks, not equality at every possible probability. The historical result demonstrates that repeated values need no separately hardcoded atom capacity.

== Current Rust K32 evidence

The current `backend_accuracy` executable uses 100,000 samples at each of 32 representative tensor positions, nine query probabilities from 0.001 through 0.999, and tie-aware empirical rank intervals. RankKnot and Quantile Spine use their default 256-row buffers; t-digest uses its default 200-row buffer. Its adversarial suite uses 65,536 samples at one position and 1,003 probabilities. Heap figures come from an instrumented global allocator and exclude inputs, exact truth, and query outputs.

#table(
  columns: (1.25fr, 1fr, 1fr, 1fr),
  align: (left, right, right, right),
  table.header([*Workload*], [*K32 mean / max*], [*t-digest mean / max*], [*Spine mean / max*]),
  [normal], [0.000458 / 0.001460], [0.000627 / 0.006950], [0.000302 / 0.001490],
  [uniform], [0.000681 / 0.001170], [0.001334 / 0.007380], [0.000575 / 0.001780],
  [lognormal], [0.000860 / 0.002290], [0.001831 / 0.012570], [0.000758 / 0.002130],
  [32-level normal], [0.000094 / 0.000660], [0.009511 / 0.034780], [0.011590 / 0.055840],
  [50% zeros], [0.000380 / 0.001380], [0.026949 / 0.252720], [0.000096 / 0.000600],
  [95% zeros], [0.000508 / 0.006160], [0.000359 / 0.004750], [0.000031 / 0.000420],
  [heterogeneous], [0.000511 / 0.002260], [0.003170 / 0.051750], [0.001411 / 0.039770],
)

K32 substantially improves on t-digest for most representative workloads, especially quantized and 50%-zero data, but it is not uniformly best. Quantile Spine has lower mean error on several smooth distributions and clearly wins on 95%-zero activations. K32 often has the smallest maximum error and is particularly strong on the heterogeneous workload. Adversarial results are mixed, so no universal ordering is claimed.

Allocator measurements at 32 positions were 39,432 live / 45,064 peak bytes for K32, 182,408 / typically 239,208 for t-digest, and 77,320 / 113,160 for Quantile Spine. At one position K32 measured 1,240 live / 6,872 peak bytes; the difference between live and peak is worker-local ingestion scratch.

A local Apple M4 Divan run with RankKnot’s default capacity 256 and each comparison backend’s default configuration measured median update times of 8.73 ms for K32 versus 7.72 ms for t-digest on 64×64×1,000 normal data, 8.81 versus 7.51 ms on the corresponding uniform data, and 21.47 versus 26.81 ms on 256×256×200 uniform data. K32 therefore remained 13–17% slower on the smaller repeated-flush cases but was about 20% faster on the larger one. These are platform-specific observations, not portable guarantees.

== Important caveats

- *Historical extreme tails:* in the K64 study, t-digest was often two to three times more accurate at individual smooth tail queries, although K64’s absolute tail errors were small.
- *Separated modes:* linear interpolation across a large unsupported value gap remains a principal failure mode.
- *Extrema:* current K32 endpoint requests use exact min/max sidecars; the historical K64 interior results did not include endpoint requests.
- *Throughput scope:* the Apple M4 timings above are local medians. Different tensor widths, Rayon pools, CPUs, and system load can change the comparison.
- *No consumer evidence:* no RankKnot quantizer, group merge, alert, visualization, or serialization study exists because those APIs are not implemented.

= A deliberately small systems design

== Rust representation

```rust
const RANK_KNOT_K: usize = 32;

#[repr(C)]
struct RankKnotState {
    values: [f32; RANK_KNOT_K],
    masses: [u16; RANK_KNOT_K],
    pure_mask: u64,
    min: f32,
    max: f32,
}
```

`RankKnotState` is crate-private; the public `RankKnot` type is a zero-sized kernel marker. One `u64` sample count is shared across positions in `RankKnotStorage`. Temporary weights and weighted moments use `u64` and `f64`; persistent representatives round to `f32`. With 256 buffered rows, tensor-scaled retained storage is approximately 208 + 1,024 = 1,232 bytes per position. Worker scratch and object headers are additional.

== One current ingestion reducer

The deterministic reducer currently implements buffered ingestion only. Reusing it for cold merges remains a design direction. Current RankKnot callers cannot access the encoded support, and no RankKnot-specific cache or query-time workspace is retained per position.

== Minimal configuration

#block(breakable: false)[
  #table(
    columns: (1.35fr, 1fr, 2.1fr),
    table.header([*Item*], [*Current value*], [*Status*]),
    [`buffer_capacity`], [`256`], [public and configurable; must be positive],
    [input type], [`f32`], [RankKnot is implemented only for `f32`],
    [knot count], [`32`], [fixed internal constant],
    [mass quanta], [`65,535`], [fixed internal `u16` normalization],
    [rank scale], [arcsine], [fixed internal targets],
    [NaN], [unsupported], [not validated on update; panics during flush],
    [infinities], [supported], [protected pure singleton groups and exact endpoints],
  )
]

Only `buffer_capacity` is public configuration. The mass encoding, knot count, and scale function remain implementation details while long-stream and cross-platform behavior stabilizes.

= Validation status

== Implemented coverage

#table(
  columns: (1.25fr, 2.75fr),
  table.header([*Area*], [*Current evidence*]),
  [Buffering], [default/custom capacity, partial and full flushes, and sample count],
  [Queries], [monotone curves, NaN query probability, exact endpoints, and tensor/cell paths],
  [Discrete data], [retained tie intervals, constants, and deterministic signed-zero extrema],
  [Exceptional values], [protected positive and negative infinities],
  [Encoding], [ordered support, active masses summing to 65,535, and ties-to-even quantization],
  [Accuracy/memory], [`backend_accuracy` representative and adversarial suites with allocator instrumentation],
  [Throughput], [Divan benchmarks for normal and uniform tensors at two representative sizes],
)

== Open work

- define and test an atomic NaN policy, or make the NaN-free precondition part of the stable contract;
- test requantization drift through at least one million updates and repeated shifts;
- check in reproducible current K32 accuracy and benchmark result artifacts with platform metadata;
- replace the RankKnot and Quantile Spine `merge_*` panic stubs with implementations and fidelity tests;
- add RankKnot state export or serialization before claiming durable deferred analysis;
- build visualization, PTQ, or threshold consumers and measure application outcomes;
- continue throughput work on smaller repeated-flush tensors, where K32 still trails t-digest locally.

Ascending streams, blocked modes, and crafted float patterns remain useful diagnostics rather than universal gates. Fixed memory does not provide a worst-case distribution-free accuracy guarantee.

= Risks and limitations

- *Approximation, not archival:* individual observations and temporal order are irrecoverable, and RankKnot has no snapshot format.
- *Representative data:* no summary can infer a deployment regime that was never observed.
- *Marginal state:* cross-position covariance and sample identity are absent.
- *Requantization drift:* 16-bit masses compactly encode 32 locations but repeatedly approximate old probability.
- *Shared count and NaN:* NaN currently reaches the buffer and panics during flush. Per-position missingness or atomic rejection requires revised semantics or extra work.
- *Tail and mode parity:* no backend wins every workload; sparse zero activations, extreme tails, and separated modes remain important counterexamples.
- *Working memory:* the 208-byte title names state only. The default input buffer raises tensor-scaled retained storage to about 1,232 bytes per `f32` position, and worker scratch raises peak heap.
- *Unavailable consumers:* merge, analysis, zero-filtering, serialization, and visualization methods are contract stubs that panic for RankKnot; knots are crate-private; PTQ and direct support export are absent.
- *Fixed memory:* no 208-byte state offers tiny distribution-free error for every stream.

#callout([The design principle], [Collection should preserve a useful approximation of the observed distribution while making as few downstream choices as possible. RankKnot spends its state budget on rank mass, ties, and extrema; everything else is deferred.], tone: green)

= Reproducibility

Current K32 Rust behavior is defined and exercised by:

- `monatq/src/kernels/rankknot.rs`
- `monatq/tests/rankknot.rs`
- `monatq/src/bin/backend_accuracy.rs`
- `monatq/benches/tensor_digest.rs`

Run `cargo run -p monatq --release --bin backend_accuracy` for the current accuracy/memory report and `cargo bench -p monatq --bench tensor_digest` for throughput. The numeric K32 results in this note came from a local Apple M4 run and are not yet checked in as a machine-readable artifact.

Historical K64 feasibility evidence remains in:

- `docs/experiments/transport_coreset_feasibility.py`
- `docs/experiments/results/transport-coreset-holdout.csv`
- `docs/experiments/README.md`

That script simulates 48-location `u32` and 64-location `u16` layouts and invokes repository backends through a temporary Rust crate. Its unqualified “RankKnot” label means the historical K64 prototype, not current K32.

= Conclusion

Current RankKnot is a working K32, `f32`-only quantile backend. Its 208-byte per-position state preserves exact extrema and retained pure ties, its default live allocation is substantially smaller than both comparison backends, and current Rust accuracy is generally stronger than t-digest on the representative suite without being uniformly best. Local throughput beats t-digest on the larger tested tensor and still trails it on smaller repeated-flush cases.

The broader K64 Python results remain useful evidence for the design’s capacity trade-off, but they are not current-backend results. Merge, analysis, zero-filtering, serialization, and visualization are present in the generic contract but remain panic stubs for RankKnot; direct knot access, clipping search, and quantization are absent. The next decisions should be driven by long-stream drift, reproducible cross-platform throughput, an explicit NaN contract, and real consumer studies rather than by the historical prototype alone.

#bibliography("references.bib", title: [References])
