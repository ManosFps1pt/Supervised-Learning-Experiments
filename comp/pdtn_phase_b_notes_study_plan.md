# PDTN Phase B Notes Study Plan

This file is not for learning new material.

It is for learning how to use the notes fast under pressure.

The goal for the last 2 evenings is:

- stop searching randomly
- know where each kind of problem lives in the notes
- find the right snippet quickly
- apply a baseline or fix without hesitation

## 1. Main Principle

Treat the notes like a tool, not a textbook.

Your goal is not:

- read everything again
- memorize every code block
- improve the notes again

Your goal is:

- memorize the map
- practice retrieval
- build pressure habits

## 2. The Map You Should Know By Memory

Memorize these sections first:

- `2`: first baseline by problem type
- `4`: sklearn / tabular fixes
- `6`: PyTorch errors, dtype, device, shapes, loss issues
- `7`: vision, transforms, channels, CNN reminders
- `8`: text, embeddings, TF-IDF, cosine similarity
- `9`: optimization / solver problems
- `10`: debugging checklist
- `11`: worked problems from this repo

If you know this map, you will already save time.

## 3. Fast Trigger -> Section Lookup

Use these trigger words during competition:

- `baseline` -> Section `2`
- `tabular / sklearn / NaN / encoding` -> Section `4`
- `dtype / device / shape / loss / DataLoader / batch` -> Section `6`
- `images / transforms / channels / pretrained backbone` -> Section `7`
- `TF-IDF / retrieval / cosine / embeddings / token ids` -> Section `8`
- `constraints / assignment / scheduling / discrete optimization` -> Section `9`
- `I do not know what is broken` -> Section `10`
- `this looks like an old solved repo problem` -> Section `11`

## 4. Competition Habit

When blocked, do this in order:

1. identify the task type
2. jump to the matching baseline section
3. implement the cheapest correct baseline
4. validate locally
5. if broken, jump to the matching error/debug section
6. only improve after a correct baseline exists

If you are stuck for more than 2 minutes:

1. classify the problem
2. open the matching section
3. copy the baseline pattern
4. use the debugging checklist

## 5. What To Practice

Do not just reread.

Practice lookup.

Use prompts like:

- tabular classification baseline?
- regression loss and target dtype?
- CrossEntropy target dtype?
- stack vs cat?
- binary classification loss?
- embedding retrieval normalization?
- DataLoader worker crash?
- submission sanity check?
- grayscale image into RGB backbone?
- coordinate-to-RGB regression like squarepainting?

For each prompt:

- start from the top of the notes
- find the correct section fast
- find the exact snippet or rule
- explain in one sentence why that section is the right one

## 6. Two-Evening Plan

### Evening 1

Goal:

- memorize the structure of the notes

Steps:

1. skim the notes once for section structure only
2. write the section map from memory
3. review the section map until you can reproduce it
4. do 5 timed retrieval drills

Timed retrieval drill format:

- one prompt
- 30 to 60 seconds
- locate the correct section
- locate the relevant snippet
- say out loud what it is for

### Evening 2

Goal:

- simulate contest retrieval under pressure

Use 3 to 5 mini-scenarios such as:

- image classifier not learning
- classification labels wrong dtype
- text task with no pretrained weights
- squarepainting-style RGB regression
- solver problem vs ML problem

For each scenario:

- start a timer
- use only the notes
- find the matching sections
- write or describe the first baseline or fix
- stop after 10 to 15 minutes

You are training navigation speed, not perfect solutions.

## 7. Five Emergency Queries To Memorize

These should become automatic:

- wrong task / wrong loss pairing
- shape mismatch
- device mismatch
- NaN / OOM
- submission / export mistake

If one of these appears, you should already know where to go.

## 8. What Not To Do

- do not passively reread the full notes again and again
- do not try to memorize every code block
- do not spend the last 2 evenings expanding the notes
- do not practice only when calm; use a timer
- do not let yourself browse aimlessly through sections

## 9. Best Mental Model

The notes are a decision system:

- identify task
- find baseline
- find error section if needed
- find similar worked problem if needed
- submit a correct version early

That is better than having bigger notes but no retrieval speed.

## 10. Five-Minute Pre-Competition Review

Right before the competition, review only this:

- the section map
- the trigger -> section mapping
- the 2-minute stuck rule
- the 5 emergency queries
- the baseline-first principle

Do not start deep studying at the last minute.

## 11. One-Line Rule

If I am stuck, I do not search randomly.

I classify the problem, jump to the matching section, use the cheapest correct baseline, and debug with the checklist.
