randomArray ← {?⍵⍴0}

⍝ the box-muller transform is used to generate normally distributed random numbers from uniformly distributed numbers
normalArray ← {
    ⍝ ⍵ is the shape of the array
    u1 ← randomArray ⍵
    u2 ← randomArray ⍵

    sqrt ← (¯2×⍟u1)*0.5
    cos ← 2○2×○u2

    sqrt×cos
}


⍝ visually verifying that normalArray actually produces normally distributed random numbers
makeHistogram ← {
    ⍝ ⍺ is the optional scale (defaults to 1)
    ⍝ ⍵ is the array of numbers to graph

    ⍺ ← 1
    scale ← ⍺

    ⍝ from aplcart.info
    histogram ← (↑'⎕'⍴¨⍨⌊)

    nbins ← 20

    ⍝ 20 bins from -5 to +5
    bins ← 2÷⍨¯10+⍳nbins

    binned ← bins⍸⍵

    ⍝ https://aplwiki.com/wiki/Key#Vocabulary
    ⍝ concatenating ⍳bins in the beginning makes sure all bins are ordered and defined
    ⍝ the function to the left of ⌸ takes two arguments:
    ⍝   ⍺ is the key (in this case the bin number)
    ⍝   ⍵ is the occurrences of the key in the array
    ⍝ so ¯1+≢⍵ counts the occurrences of that bin and removes one to account for the order initializer iota
    tallies ← {⌊scale÷⍨¯1+≢⍵}⌸(⍳nbins),binned

    histogram tallies
}


initBiasVectors ← {
    shapes ← 1↓⍵
    (shapes*0.5) ÷⍨ ⍪∘normalArray¨ shapes
}

initWeightMatrices ← {
    shapes ← 2 ,⍨/ ⍵
    sqrtCount ← 2 ×/ ⍵ * 0.5 ⍝ for numerical stability
    matrices ← normalArray¨ shapes
    matrices ÷ sqrtCount
}

initNetwork ← {
    ⍝ ⍵ is the shape of the network
    ⍝ returns a pair (W b) of weight matrices and bias vectors

    (initWeightMatrices ⍵) (initBiasVectors ⍵)
}
