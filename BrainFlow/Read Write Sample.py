import time

import numpy as np
import pandas as pd
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter


def main():
    BoardShim.enable_dev_board_logger()

    # use cyton daisypip board for demo
    params = BrainFlowInputParams()
    params.serial_port = "COM3"
    board = BoardShim(BoardIds.CYTON_DAISY_BOARD, params)

    
    board.prepare_session()
    board.start_stream()
    BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'start sleeping in the main thread')
    sampling_rate = board.get_sampling_rate(BoardIds.CYTON_DAISY_BOARD,0) # I added this to see the saampling rate
    print("Sampling Rate:")
    print(sampling_rate)
    time.sleep(10)

    data = board.get_board_data()

    board.stop_stream()
    board.release_session()

    # demo how to convert it to pandas DF and plot data
    eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD.value)
    df = pd.DataFrame(np.transpose(data))
    print('Data From the Board')
    print(df.head(10))

    # demo for data serialization using brainflow API, we recommend to use it instead pandas.to_csv()
    DataFilter.write_file(data, 'test.csv', 'w')  # use 'a' for append mode
    restored_data = DataFilter.read_file('test.csv')
    restored_df = pd.DataFrame(np.transpose(restored_data))
    print('Data From the File')
    print(restored_df.head(10))


if __name__ == "__main__":
    main()