#let config(
  title: none,
  authors: (),
  abstract: none,
  keywords: (),
  doc,
) = {

  set page(
    paper: "a4",
    header: align(right)[
      CMA-ES
    ],
    numbering: "1",
  )

  set par(
    justify: true,
  )

  set text(font: "CMU Serif")

  set heading(numbering: "1.")

  set cite(style: "chicago-author-date")

  set math.equation(numbering: "(1)")

  // Reference style
  set ref(supplement: it => {
    let fig = it.func()
    if fig == math.equation {
      "Eq."
    }
    
    else if fig == figure {
      it.supplement
    }
  })

  // Title
  align(center, text(16pt)[
    #title
  ])

  // Authors
  if authors.len() == 0 {
      align(center, text(14pt)[
        Gaëtan Serré \
        ENS Paris-Saclay - Centre Borelli \
        #text(font: "CMU Typewriter Text")[
          #link("mailto:gaetan.serre@ens-paris-saclay.fr")
        ]
      ])
  } else {
    for author in authors {
      align(center, text(14pt)[
        #author.name \
        #author.affiliation \
        #text(font: "CMU Typewriter Text")[
          #link("mailto:" + author.email)
        ]
      ])
    }
  }

  // Abstract
  let width_box_abstract = 80%

  if abstract != none {
    align(center, text()[*Abstract*])
    align(center, 
      box(width:width_box_abstract, 
        align(left, text(size: 10pt)[
          #abstract
        ])
      )
    )
  }
  
  // Keywords
  align(center, box(width:width_box_abstract,
    align(left, {
      set text(size: 10pt)
      if keywords.len() > 0 {
        [*Keywords: *]
        let last_keyword = keywords.pop()
        for keyword in keywords {
          [#keyword] + [; ]
        }
        [#last_keyword.]
      }
    })
  ))
  
  // Indentation
  set par(
    first-line-indent: 1em
  )

  doc
}

// Math environment

#let heading_count = counter(heading)

#let math_block(supplement, counter, name, it) = {
  counter.step()
  if name == "" {
    block([*#supplement #heading_count.display()#counter.display().* ] + emph(it))
  } else {
    block([*#supplement #heading_count.display()#counter.display() * (#name). ] + emph(it))
  }
}

// Counters

#let th_count = counter("theorem")
#let theorem(name, it) = math_block("Theorem", th_count, name, it)

#let def_count = counter("definition")
#let definition(name, it) = math_block("Definition", def_count, name, it)

#let lemma_count = counter("lemma")
#let lemma(name, it) = math_block("Lemma", lemma_count, name, it)

#let prop_count = counter("proposition")
#let proposition(name, it) = math_block("Proposition", prop_count, name, it)

#let proof(it) = {
  [_Proof._ $space$] + it + align(right, text()[$qed$])
}