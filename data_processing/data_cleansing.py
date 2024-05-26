import numpy as np
import pandas as pd


def HotPixel_cleansing(
        xEvent: np.ndarray,
        Timestamp: np.ndarray,
        yEvent: np.ndarray,
        polarities: np.ndarray
):
    """
    :param xEvent: Event data of x (ndarray)
    :param Timestamp: Event data of time (ndarray)
    :param yEvent: Event data of y (ndarray)
    :param polarities: The polarities of event (ndarray)
    :return: filtered event data (ndarray)
    This function is used to brush the HotPixel of data via 3-sigma rule.
    """
    df = pd.DataFrame({'Timestamp': Timestamp, 'xEvent': xEvent, 'yEvent': yEvent, 'polarities': polarities})
    df['coords'] = list(zip(df['xEvent'], df['yEvent']))
    grouped = df.groupby('coords')
    event = grouped.agg(activity=('coords', 'size'),
                        continuity=('Timestamp', lambda x: np.mean(np.diff(sorted(x))) if len(x) > 1 else 0))
    act_mean = event['activity'].mean()
    act_std = event['activity'].std()
    cont_mean = event['continuity'].mean()
    cont_std = event['continuity'].std()
    event_filtered = event[
        (event['activity'] >= act_mean - 3 * act_std) & (event['activity'] <= act_mean + 1.5 * act_std) &
        (event['continuity'] >= cont_mean - 3 * cont_std) & (event['continuity'] <= cont_mean + 1.5 * cont_std)]
    filtered_events = df[df['coords'].isin(event_filtered.index)]
    return (
        filtered_events['xEvent'].to_numpy(),
        filtered_events['Timestamp'].to_numpy(),
        filtered_events['yEvent'].to_numpy(),
        filtered_events['polarities'].to_numpy()
    )