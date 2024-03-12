import numpy as np
import pandas as pd
import logging

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

PENALTY_N_WRONG_HEXBUGS = -10_000  # only applies if the amount of predicted hexbugs is between 1 and 4
PENALTY_MORE_THAN_FOUR_HEXBUGS = -10_000
PENALTY_WRONG_NUMBER_FRAMES = -10_000
#Rings in Pixel
RINGS=          [1,  5 ,10,15,20,25]
POINTS_PER_RING=[100,50,25,15,10,5]
STREAK_POINTS= [0,2,40,60,200]
def get_score_fct(path_to_prediction: str, path_to_gt: str, log: bool = False) -> int:
    """
    Calculate the score for the given prediction and ground truth files.
    :param path_to_prediction: Path to the prediction .csv file.
    :param path_to_gt: Path to the ground truth .csv file.
    :param log: Boolean to indicate if the log should be written to a file.
    :return: Score for the given prediction and ground truth files.
    """
    final_score = 0

    if log:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(path_to_prediction.replace(".csv", "_log.log"))
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)

    # Load the data
    pred_df = pd.read_csv(path_to_prediction, index_col=0)
    gt_df = pd.read_csv(path_to_gt, index_col=0)

    # check if the predictions dataframe has at least one row
    if pred_df.shape[0] == 0:
        raise ValueError("The prediction DataFrame does not contain any rows.")

    # Check if DataFrames have the required columns
    required_columns = ['t', 'hexbug', 'x', 'y']
    if not _validate_dataframe_structure(pred_df, required_columns):
        raise ValueError(f"The prediction DataFrame does not have the required columns: {required_columns}")
    else:
        pass

    # Get the number of frames and check if they are equal
    n_frames_gt = gt_df['t'].max() + 1
    n_frames_pred = pred_df['t'].max() + 1
    """
    UNSICHER WIE SINNVOLL MIT EINBAUEN  
    if n_frames_gt != n_frames_pred:
        final_score += PENALTY_WRONG_NUMBER_FRAMES * np.abs(n_frames_gt - n_frames_pred)

        if log:
            logger.info(f"Penalty for wrong number of frames: "
                        f"{PENALTY_WRONG_NUMBER_FRAMES * np.abs(n_frames_gt - n_frames_pred)}")

        # set n_frames depending on which one is smaller
        n_frames = min(n_frames_gt, n_frames_pred)
    else:
        n_frames = n_frames_gt"""
    n_frames = n_frames_gt
    # Loop over all frames
    ids_for_streak = {i: [50, False, 0] for i in range(10)}# Maximum of 10 hexbugs?
    firstframe = True
    for idx in range(n_frames):
        if log:
            logger.info(f"Frame {idx}")

        # Get the data for the current frame
        frame_gt_df = gt_df[gt_df['t'] == idx]
        frame_pred_df = pred_df[pred_df['t'] == idx]

        # Check the number of hexbugs and apply penalties if necessary
        n_hexbugs_gt = frame_gt_df['hexbug'].nunique()
        n_hexbugs_pred = frame_pred_df['hexbug'].nunique()
        print(frame_pred_df['hexbug'])
        #used to connect ids with the entries of the distance matrix
        ids_gt = frame_gt_df['hexbug']
        ids_pred = frame_pred_df['hexbug']

        if n_hexbugs_pred > 4:
            # To prevent submissions with a lot of random guesses
            final_score += PENALTY_MORE_THAN_FOUR_HEXBUGS
            if log:
                logger.info(f"Penalty for more than four hexbugs: {PENALTY_MORE_THAN_FOUR_HEXBUGS}")
        else:
            final_score += (np.abs(n_hexbugs_pred - n_hexbugs_gt) * PENALTY_N_WRONG_HEXBUGS)
            if log:
                logger.info(f"Penalty for wrong number of hexbugs: "
                            f"{np.abs(n_hexbugs_pred - n_hexbugs_gt) * PENALTY_N_WRONG_HEXBUGS}")

        # Calculate the distance between the hexbugs
        distance_matrix = cdist(list(frame_gt_df[['x', 'y']].values),
                                list(frame_pred_df[['x', 'y']].values), 'euclidean')

        # Calculate the final score
        gt_hex, pred_hex = linear_sum_assignment(distance_matrix)  # Hungarian algorithm
        #print(ids_for_streak)
        for i, j in zip(gt_hex, pred_hex):
            gt_id = ids_gt.iloc[i]
            pred_id = ids_pred.iloc[j]
            if(ids_for_streak[gt_id][0] == pred_id):
                ids_for_streak[gt_id][1] = True # the streak is on
                ids_for_streak[gt_id][2] += 1
            else:
                ids_for_streak[gt_id][0] = pred_id
                ids_for_streak[gt_id][1] = False  # the streak is on
                ids_for_streak[gt_id][2] = 0

            distance = distance_matrix[i, j]
            points_recived = 1
            #Score per Ring in which the prediction is
            for intervalls in range(len(RINGS) - 1):
                if RINGS[intervalls] < distance <= RINGS[intervalls + 1]:
                    points_recived = POINTS_PER_RING[intervalls+1]
                    if(distance<1):
                        points_recived = 100
                    break

            #Look which streak
            for intervall in range(len(STREAK_POINTS) - 1):
                if STREAK_POINTS[intervall] < ids_for_streak[i][2] <= STREAK_POINTS[intervall + 1]:
                    points_recived = points_recived *(intervall+ 1)
                    break

            final_score += points_recived

            if log:
                logger.info(f"Distance between hexbug {gt_id} and "
                            f"{pred_id}: {distance_matrix[i, j]}   "
                            f"Points recived : {points_recived}   "
                            f" Hexbug on Fire: {ids_for_streak[gt_id][1]}")
        if log:
            logger.info("")

    if log:
        logger.info(f"\nFinal score: {final_score}")
        logger.removeHandler(handler)
        handler.close()

    return final_score


def _validate_dataframe_structure(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Validate if a Pandas DataFrame has the required columns.

    Args:
        df (pd.DataFrame): The DataFrame to validate.
        required_columns (list): A list of column names that should be present in the DataFrame.

    Returns:
        bool: True if the DataFrame has the required columns, False otherwise.
    """
    # Get the columns present in the DataFrame
    columns = df.columns.tolist()

    # Check if all required columns are present in the DataFrame
    if set(required_columns).issubset(columns):
        return True
    else:
        return False


# define main function call
if __name__ == "__main__":
    # extract args
    # import argparse
    #
    # parser = argparse.ArgumentParser()
    # parser.add_argument("path_to_prediction", help="path to the prediction file")
    # parser.add_argument("path_to_gt", help="path to the ground truth file")
    # parser.add_argument("--log", help="log the score", action="store_true")
    # args = parser.parse_args()
    # path_pred = "predicted_data_for_testing_scorecalc.csv"
    # path_test = "test_data_csv/test001.csv"
    path_pred = "test_score/test_onegoes_samecomes.csv"
    path_test = "test_score/ground_trouth.csv"
    #print(f"Score: {get_score(args.path_to_prediction, args.path_to_gt, args.log)}")
    print(f"Score: {get_score_fct(path_pred, path_test, log = True)}")

