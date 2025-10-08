from recording.data_collector import LiveDataCollector
from util.faceparser_wrapper import load_faceparser_model

load_faceparser_model()
live_data_collector = LiveDataCollector()
live_data_collector.start_collector()