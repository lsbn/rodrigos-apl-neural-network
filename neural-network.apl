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

    nbins ← 21

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


:Namespace LeakyReLU
    leaky ← 0.1

    F ← {
        ⍝ this is the activation function that introduces non-linearity between the neuron layers
        ⍵⌈leaky×⍵
    }

    dF ← {
        ⍝ derivative of the leakyReLU function
        leaky⌈×⍵
    }
:EndNamespace


_forwardPass ← {
    ⍝ ⍺⍺ is the activation function namespace
    ⍝ ⍺ is the neural network
    ⍝ ⍵ is the input to the network
    (Ws bs) ← ⍺
    (_ _ xs) ← (⍺⍺ _forwardStep⍣(≢Ws)) Ws bs (⊂⍵)

    xs
}

_forwardStep ← {
    ⍝ ⍺⍺ is the activation function namespace
    ⍝ (Ws bs xs) ← ⍵ are the components for the step
    (Ws bs xs) ← ⍵

    W ← ⊃Ws
    b ← ⊃bs
    input ← ⊃⌽xs

    x ← ⍺⍺.F b+W+.×input

    (1↓Ws) (1↓bs) (xs,⊂x)
}

⍝ network ← initNetwork 3 6 2
⍝ network (LeakyReLU _forwardPass) ⍪1 0 0
⍝ ┌─┬──────────────┬─────────────┐
⍝ │1│ 0.02095352376│¯0.0975408996│
⍝ │0│¯0.04342650119│ 0.8419090114│
⍝ │0│¯0.01685866144│             │
⍝ │ │ 0.300766094  │             │
⍝ │ │¯0.01532197623│             │
⍝ │ │ 0.07164327109│             │
⍝ └─┴──────────────┴─────────────┘


:Namespace MSELoss
    ⍝ mean square error loss function used to evaluate how far off the output of the network is from the target
    F ← {
        ⍝ ⍺ is the expected target
        ⍝ ⍵ is the network output

        sq ← 2*⍨ ⍵-⍺
        ⍝ +/, === +⌿ - this is needed because the NN outputs column vectors
        (≢sq)÷⍨ +/,sq
    }

    dF ← {
        ⍝ derivative of the MSE loss function
        ⍝ ⍺ is the expected target
        ⍝ ⍵ is the network output
        (≢⍵)÷⍨2×⍵-⍺
    }
:EndNamespace


