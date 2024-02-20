# traco_annotation

Files:
get_score: gets two csv and then computes the score for it.
files with .traco are json files
sampling: ?

traco: ?

Folders: 
files:
leaderboard data: annotations for evaluation
test: data for testing
training: data for training

Nootbooks: 
get-score: calculates score for .traco files. So json files
csvScore: calculates score for .csv files
reference-solution-ankilab: same as example-solution in traco_external but Unet is build different and postprocessing is different


Idea: gtcsv and predcsv
- Load hex IDs from the predcsv and connect them to nearest hex ID in gtscv.
- Use Distance metric to match in first frame. If there are not matching numbers of hexbug its a problem
- From there two possible solutions:
  - Do not track shortest distance anymore and only look at distance between the IDs. 
  - OR somehow try to see if the IDs have changed and then give a penalty.
- Two ideas: 
  - Use gtcsv like a target and the nearer the pred from predcsv is the more score one gets. You want to have a high one
    - One could determine the how far away it was for different ranges. So how often was it in a range of 12 pixles. 
    - Like a target with bow and arrow.
    - e.g. in a range of 5 pixles one gets 10 points. From 5-10 one gets 5 points and so on.
    - So we define a theshold and the more away the pred is the less points. 
    - The head of the hex has a certain radius - f.e. 5 pixles so everything in 5pixels from gt is considered a correct match
  - The gtcsv is like a point and the distance is the score which gets summed up. You want to have a low one. 
- Other scoreing metrics could be to see how often the model doesn't find the total number of hexbugs.:
  - Accuracy: Calculate the percentage of correctly tracked hexbugs over the total number of hexbugs in the ground truth.
  - Precision: the proportion of correctly tracked frames out of all frames where a hexbug was predicted.

- Probelm of not matching numbers of hexbugs:
  - When this encounters there should be a penalty. 
  - And also the IDs have to be connected again to see which hexbug misses. 
  - Penalty could be inrceased if the same hexbug misses more often than one time.
  - Maybe the IDs of the hex change when their heads bump together. Possible fix see above.
  - 
