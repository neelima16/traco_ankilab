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


get score:
 - if you want to have a log for your score calculation set log = True
 - if you want to have a video of you score calculation set vid = True
   -> csv must be same number of frames and there has to be a mp4 video in the same folder with the groud trouth