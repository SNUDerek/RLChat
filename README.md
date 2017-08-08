# chatbot demos with reinforcement learning

## demo 1: Q-Table chatbot demo

**slot-filling state representation**:

slots are:
- greeting (*= you gave a greeting*)
- product (*= you found out the product*)
- query (*= you found out the issue e.g. want a refund, exchange etc*)
- answer (*= you found out the solution to the problem*)
- anything_else (*= you asked if there was anything else*)
- goodbye (*= you said goodbye*)

states are binary 0/1 vectors of length `slots` (don't know-know)

**simple action representation**:

all actions are assumed to be info meant to elicit exactly one slot value

the `environment` is assumed to reply at 100% rate (not including noise)

so all actions and responses correspond to and are represented by one slot value.

*future update should have slot-values saved in memory*

**environment**

the environment consists of a simulated customer with a rule-based policy and a noise adder

the noise simulates miscategorizations in the NLU/NLG modules, etc.

the customer, as noted above, is assumed to answer all queries that it successfully understands.
(*noise could be added to customer input, currently not implemented*)

this approach is `bootstrapping` for later human training

this bootstrapping with simulated environments has been shown to be effective for RL
(*citation needed!*)

**noisy channel**:

noise is simulated by changing the last state update at random

**analysis**

(*NB: these numbers are out-of-date but still reflective of the results*)

the state values correspond to the following:

`['greeting', 'product', 'query', 'answer', 'anything_else', 'goodbye']`

the following are the states and associated Q-table values:

at initial state, best 'quality' (=Q) choice is *greeting* with *product* and *query* only slightly worse, whereas giving an answer, asking if there is anything else, or saying goodbye are distinctly worse

```
[0, 0, 0, 0, 0, 0] [-0.5212230282133684, # greeting
                    -0.6063718322484382, # product
                    -0.5290990055197529, # query
                    -0.9928034851793454,
                    -0.9913866056965311,
                    -0.991500257618121]
```

assuming we say a greeting, saying another greeting is strongly penalized, asking for product or problem information are the best, any the rest are also bad options

```
[1, 0, 0, 0, 0, 0] [-1.636127048557182,
                    -0.8028932057921628, # product
                    -0.6813012092487777, # query
                    -0.9926351246040384,
                    -0.9923468269316451,
                    -0.9925702810114896]
```

if the only things left are to ask `anything_else` and/or say `goodbye`, the best choices are those choice(s)

```
[1, 1, 1, 1, 0, 0] [-1.8955268663051998,
                    -1.8955268663051998,
                    -1.8955268663051998,
                    -1.8843851530616844,
                    -1.1365154651007345, # anything_else
                    -0.9950298514502222] # goodbye

[1, 1, 1, 1, 1, 0] [-1.8934962430845368,
                    -1.8934962430845368,
                    -1.8934962430845368,
                    -1.8934962430845368,
                    -1.8934962430845368,
                    -0.992773603427263] # goodbye
```
