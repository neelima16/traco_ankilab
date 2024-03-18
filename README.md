# traco_annotation
 - check number of frames
 - init ids_for_streak with [50, False, 0] with id,strak,framesofstrak for n hexbugs
 - loop over all frames
 - get gt and pred hexbugs and look for different numbers of hexbug to applie penalty
 - get distance matrix
 - get shortest distance between hexbugs to assign the ids from the pred to gt ids
 - look if ids are the same as in last frame and if not get penatly 
 - get points frome rings
 - look at what streak and double or sth
 - 


TO DO:
  - Ein repository ohne testdaten
    -> kann man einen passwortgeschützten folder anlegen?
  - Score berechnung wie gedacht.
    - dazu in den csv nach frame suchen. Also nach t und dann die connecten weil ordnung anders sein kann.



Fragen:
    - Für test_25 keine traco?
    - Streak so wie ist oder das man 