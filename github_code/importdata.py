###
# Easy Electrophysiology liscense xxxx
# original repo can no longer be accessed: https://github.com/easy-electrophysiaology/easy-electrophysiology-analysis
# code adapted from the repo
###


from __future__ import annotations

import neo
import numpy as np
from github_code import core_analysis_methods
from neo.rawio.baserawio import _signal_channel_dtype
 
class ImportData:

    # github code contains "cfgs: ConfigsClass, mw: MainWindow" as input argument
    # also with self.cfgs = cfgs and self.mw = mw
    # but I don't need the gui settings and the window, so I deleted these two lines

    # when I called "importer = ephys.importdata.ImportData(filepath, ext)" in my batch.py
    # I call the __init__ function to instantialize my object
    # i.e I created a new object with attributes "full_filepath", "ext", "channels", "reader", and "neo_block"
    # P.S "channels", "reader", and "neo_block" was the output by calling "load_file_with_neo"
    
    # this new object is assigned to a variable named "importer"
    # I also have 13 methods including "load_data", "load_common_neo_api_files", "load_file_with_neo",
    # "get_channel_data_info","get_user_input_channel_units","get_default_channels",
    # "make_channels_from_analogsignals","channel_is_proxy",
    # "channel_2_exists_and_is_different_type_to_channel_1","get_channel_type","get_channel_type_from_channel_units",
    # "show_bad_header_units_warning", "check_if_header_units_are_correct"
    # I can call these methods on my new object "importer" by object.load_data()

    # data = importer.load_data()
    # So I called the load_data() method on my newly created object "importer"
    # And I assigned the output to a variable named "data"
    


    def __init__(self, full_filepath, ext):
        self.full_filepath = full_filepath
        self.ext = ext.upper()
        self.channels, self.reader, self.neo_block = self.load_file_with_neo()

    def load_data(self):
        # I deleted the if line because I think my channels won't be None
        #if self.channels is None:
        #    return

        channel_1,channel_1_idx,channel_2,channel_2_idx = self.get_channel_data_info() # I removed the parenthness to make it easier to read

        #if (
        #    channel_1 is None
        #    or channel_1_idx is None  # type narrow
        #):
        #    return

        # Get channel type (Im or Vm), units and load data structure
        channel_1_units, channel_1_type = self.get_channel_type(channel_1, channel_1_idx)

        # if the channel_1_type is empty, exit this function ealier
        if not channel_1_type:
            return

        if channel_2 and channel_2_idx is not None and not self.channel_is_proxy(channel_2_idx): # if channel 2 exists and the channel is real
            channel_2_units, channel_2_type = self.get_channel_type(channel_2, channel_2_idx)

            if not channel_2_type:
                return


            Data = RawData(
                self.neo_block,
                2,
                channel_1_type,
                channel_1_units,
                channel_1_idx,
                channel_2_type,
                channel_2_units,
                channel_2_idx,
                self.reader,
                self.channels,
                self.ext,
            )
        else:
            Data = RawData(
                self.neo_block,
                1,
                channel_1_type,
                channel_1_units,
                channel_1_idx,
                None,
                None,
                None,
                self.reader,
                self.channels,
                self.ext,
            )
        return Data


    # --------------------------------------------------------------------------------------
    # Load Data With Neo
    # --------------------------------------------------------------------------------------
    # Methods for loading in data. Generates a "Data" class that contains Vm, Im,
    # and Time data. Uses the neo electrophysiology module. Handles 1, 2 or more
    # channels (but forces user to input 2 channels when there are > 2. The main
    # method "import_data" will return None if import fails.
    # --------------------------------------------------------------------------------------


    def load_common_neo_api_files(self):

        # Load the reader
        # Because I know my file is .abf
        # I deleted the elif to process AXGX, AXGD, WCP, EDR, H5, SMR, and IBW
        if self.ext == ".ABF":
            reader = neo.AxonIO(self.full_filepath)
        else:
            return None, None, None

        if self.ext in [".H5", ".IBW"]:
            neo_block = reader.read_block()
        else:

            neo_block = reader.read_block(signal_group_mode="split-all")

        if self.ext == ".IBW":
            channels = self.make_channels_from_analogsignals(neo_block)
            if channels is False:
                return None, None, None
        else:
            channels = reader.header["signal_channels"]

        return channels, reader, neo_block



    def load_file_with_neo(self):

        # because i knew my data is not with .DAT or .CSV extension, I deleted the first two lines if/else from the code of github
        # and I call the "load_common_neo_api_files()" directly, instead of waiting until the extension is not .DAT or .CSV then I call
        channels, reader, neo_block = self.load_common_neo_api_files()

        # if the channel was empty, I return none for both the channels, reader, and neo_block
        # this is important because in the "load_data()" method, I will first check if "self.channels is None"
        # and if it is None, I will then run "get_channel_data_info()" inside my "load_data()" later
        if channels is None:
            return None, None, None

        return channels, reader, neo_block




    # Processing Channels
    # ----------------------------------------------------------------------------------

    # for the method "get_channel_data_info()", the original Github code has
    # if self.cfgs.file_load_options["select_channels_to_load"]["on"]:
    #     return self.get_users_default_channels()
    # meaning that, if there were more than 2 channels, used could select which channels to load
    # then the code call "get_users_default_channels() instead of "get_default_channels"
    # In my pipeline I always have two channels to load only, at least I think
    # And I don't want user inputs
    # so I deleted the if conditioning, and instead run and return the output of "get_default_channels" on my "importer" instead
    
    def get_channel_data_info(self):
        return self.get_default_channels()

    def get_user_input_channel_units(
        self, channel_units, channel_idx):
        return "mV", "Vm" # return mV as channel_units
        ## originally,
        # the "get_user_input_channel_units(channel[4].strip(), channel_idx)is accepting the raw channel unreocgnized text (e.g mN) and the index as input
        # then, the Github code use util.get_user_input() to ask user to input a value
        # this value is assigned as the variable channel_type
        # but this value has a restriction, the units must be one of these fA, pA, nA, uA, mA, A, pV, nV, uV, mV, V
        # if the user enters correctly, this channel_units is then taken as input,
        # to re-run the function "get_channel_type_from_channel_units(channel_units, channel_idx)
        # because our data of mechanical stimulator is mN and we don't process this unit, but we don't need this data anyway, so unit does not matter
        # I force return this user input as mV




    def get_default_channels(self):
        # First assume the channel 2 is none, then update this variable if channel 2 exist
        # This is to prevent error from popping up if the data only have 1 channel
        channel_2 = None # change from Github "channel_2 = channel_2_idx = None" to make it easier to understand
        channel_2_idx = None

        # assert self.channels is not None, "Type Checking, self.channels cannot be None"
        num_channels = len(self.channels) # as we shown above, channels variable is a nested list, printing the first channel, then the seconds etc

        if num_channels == 1 or num_channels == 2: # change from original line "if num_channels in [1, 2]:" to make the line more understandable by me
            channel_1_idx = 0
            # we assign the channel 1 index as 0,
            # this index is used to subset the "channels" variable,
            # which was a list that we created inside our "load_common_neo_api_files()"
            # when we called "channels = reader.header["signal_channels"]"
            # so "self.channels[channel_1_idx]" actually contains
            # ('VClamp', '0', 20000., 'int16', 'pA', 0.15258788, 0., '0', '0')
            channel_1 = self.channels[channel_1_idx]

            if num_channels == 2:
                # if we have two channels, now go to the channel 2 to get the information
                channel_2_idx = 1
                channel_2 = self.channels[channel_2_idx]
                # which is  ('IN5', '1', 20000., 'int16', 'V', 0.00030518, 0., '0', '0')
        elif num_channels > 2:
            # if there are more than 2 channels, sorry my program is too basic we only process two channels, so process the first two anyway
            channel_1_idx, channel_2_idx = 0, 1
            channel_1 = self.channels[channel_1_idx]
            channel_2 = self.channels[channel_2_idx]
        else:
            raise TypeError("num channels must be > 0")

        return channel_1, channel_1_idx, channel_2, channel_2_idx  # type: ignore



    def make_channels_from_analogsignals(self, neo_block) -> Union[Literal[False], NDArray]:
        """ """
        num_chan = len(neo_block.segments[0].analogsignals)

        all_channels = []

        for chan_idx in range(num_chan):
            assert neo_block.segments[0].analogsignals[0].units.magnitude == 1
            units = neo_block.segments[0].analogsignals[0].units.dimensionality.unicode
            fs = neo_block.segments[0].analogsignals[0].sampling_rate.magnitude

            all_channels.append(
                (str(chan_idx), str(chan_idx), fs, None, units, None, None, None),
            )

            # Check that all records have the same units / sampling rate
            if not len(neo_block.segments) == 1:
                self.mw.show_messagebox(
                    "Load File Error",
                    "Multi-segment records not currently supported. Please contact" "support@easyelectrophysiology.com",
                )
                return False

            for seg in neo_block.segments:  # For future...
                units_same = units == seg.analogsignals[chan_idx].units.dimensionality.unicode
                fs_same = fs == neo_block.segments[0].analogsignals[0].sampling_rate.magnitude

                if not (units_same and fs_same):
                    self.mw.show_messagebox(
                        "Load File Error",
                        "The file contains different units or sampling rates across "
                        "records. This type of recording is not supported.",
                    )
                    return False

        return np.array(all_channels, dtype=_signal_channel_dtype)

    # Processing Channels
    # ----------------------------------------------------------------------------------

    # this function check if the channel is really existing
    # I removed the following block so that we don't have to print an error message on any gui/my window
    #   if channel_1 and channel_is_proxy:self.show_warning("Load File Error","Please ensure the data channel exists in the file. ""Please contact support@easyelectrophysiology.com if ""this channel definitely exists.",)
    # other code all stays the same as GitHub
    def channel_is_proxy(self, channel_idx: Int, channel_1: bool = False) -> bool:
        assert self.neo_block is not None, "Type Check neo block"
        segment_analogsignals = self.neo_block.segments[0].analogsignals

        if channel_idx + 1 > len(segment_analogsignals):
            channel_is_proxy = True
        else:
            channel_is_proxy = not hasattr(segment_analogsignals[channel_idx], "magnitude")

        return channel_is_proxy

    def channel_2_exists_and_is_different_type_to_channel_1(
        self, channel_1_type: str, channel_2_type: Optional[str]
    ) -> bool:
        if not channel_2_type:
            return False

        if channel_1_type == channel_2_type:
            self.show_warning(
                "Load File Error",
                "Cannot have two channels of the same data type "
                "(e.g. voltage, voltage). "
                "Please load only one of the channels (File > File Loading Options > "
                "Select Channels to Load).",
            )
            return False
        return True




    def get_channel_type(self, channel, channel_idx):
        """
        Find the channel type based on it's saved name or units.

        Often the channel name reflects the input number of the amplifier
        e.g. IN0 and is no use. In this case the units is used.
        """
        
        # for 4 times stimulation (each sweep four cue, 4 sweeps (stimulate 1 time each at the 4th cue)
        # the channel_1 of channel_2 is 
        #('VClamp', '0', 20000., 'int16', 'pA', 0.15258788, 0., '0', '0')
        #('IN5', '1', 20000., 'int16', 'V', 0.00030518, 0., '0', '0')
        # when we pass this channel_1 or channel_2 into this "get_channel_type" function
        # the input argument is now renamed as "channel"
        # because we did "get_channel_type(self, channel, channel_idx)"
        # so "channel[4].strip()"
        # means pA and V for channel 1 and 2 respectively
        # .strip() means remove space
        
        channel_units = channel[4].strip()

        if channel_units == "\x02mV":
            channel_units = "mV"

        channel_units, channel_type = self.get_channel_type_from_channel_units(channel_units, channel_idx)

        if channel_units == "not_recognised":
            channel_units, channel_type = self.get_user_input_channel_units(channel[4].strip(), channel_idx)

        return channel_units, channel_type  # type: ignore

    

    def get_channel_type_from_channel_units(self, channel_units, channel_idx):
        """
        TODO
        ----
        add channel_units to configs as copied in importdata_show_channels
        """
        if channel_units == "mV":
            channel_type = "Vm"
        elif channel_units == "pA":
            channel_type = "Im"
        elif channel_units in ["fA", "nA", "uA", "mA", "A"]:
            channel_type = "Im"
        elif channel_units in ["pV", "nV", "uV", "V"]:
            channel_type = "Vm"
        else:
            return "not_recognised", None

        #print(f"[DEBUG] channel_idx={channel_idx} units='{channel_units}' -> type={channel_type}")

        return channel_units, channel_type


    # Handle non-pA or mV channel types
    # ----------------------------------------------------------------------------------

    def show_bad_header_units_warning(self, channel_type: ImOrVm, channel_units: str, channel_idx: Int) -> None:
        """
        Prompt the user to set the correct settings for re-scaling data
        that is not in pA or mV.

        In some cases (typically Axograph files) the header reads A but the
        data is scaled to nA. Here guess the most likely scaling based on the
        range of the data and prompt the user if the guessed unit type does
        not match the header unit type.
        """
        expected_units = "pA" if channel_type == "Im" else "mV"
        message_string = (
            "The units in the file header are {0} "
            "but are expected to be {1}. Please specify default "
            "file loading options in 'File > File Loading Options > "
            "Specify Default Unit Handling'.".format(channel_units, expected_units)
        )

        guessed_units = self.check_if_header_units_are_correct(channel_idx, channel_type)
        if guessed_units != channel_units:
            extra_warning = (
                "\n\nWARNING! The units in the file header are {0} but "
                "these data are most likely in {1}. Try {1} first in "
                "'File Load Options'.".format(channel_units, guessed_units)
            )
            message_string += extra_warning

        self.mw.show_messagebox("Channel Units Error", message_string)

    def check_if_header_units_are_correct(self, channel_idx: Int, im_or_vm: ImOrVm) -> str:
        """
        Typically data properly scaled to Im / mV will be in the 0-2000 range.
        The only instance this could cause problems is for very large
        stimulus registering > 2000, but these are rarely seen pA / mV.
        """
        assert self.neo_block is not None, "Type Check neo block"
        channel_data = self.neo_block.segments[0].analogsignals[channel_idx].magnitude

        channel_range = max(channel_data) - min(channel_data)

        if im_or_vm == "Im":
            unit_ranges = {
                "fA": (2000, 9999),
                "pA": (1, 1999),
                "nA": (0.001, 0.999),
                "uA": (0.000001, 0.000999),
                "mA": (0.000000001, 0.000000999),
                "A": (0.000000000001, 0.000000000999),
            }
        elif im_or_vm == "Vm":
            unit_ranges = {
                "pV": (100000, 999999),
                "nV": (10000, 99999),
                "uV": (2000, 9999),
                "mV": (1, 1999),
                "V": (0.001, 0.999),
            }

        for unit, range_ in unit_ranges.items():
            if range_[0] < abs(channel_range) < range_[1]:
                return unit
        return "undefined: contact Easy Electrophysiology."



# --------------------------------------------------------------------------------------
# Data Class
# --------------------------------------------------------------------------------------


class RawData:

    # Github code contains "cfgs: ConfigsClass" as input argument,
    # I don't need cfgs so I deleted this 
    def __init__(
        self,
        neo_block: Block,
        num_chans: Int,
        channel_1_type: str,
        channel_1_units: Optional[str],
        channel_1_idx: Int,
        channel_2_type: Optional[str],
        channel_2_units: Optional[str],
        channel_2_idx: Optional[Int],
        reader: BaseIO,
        channels: NDArray,
        file_ext: str,
    ):
        # GitHub code contains "self.load_setting = cfgs.file_load_options["force_load_options"]"
        # deleted my me as this is no longer an option 
        self.num_recs = len(neo_block.segments)
        self.num_samples = len(neo_block.segments[0].analogsignals[channel_1_idx])#e.g this means first segment first channel "len(neo_block.segments[0].analogsignals[0])"

        # strip quantities
        self.fs = neo_block.segments[0].analogsignals[channel_1_idx].sampling_rate.magnitude # use the .magnitude to extract the numeric value without unit
        self.ts = neo_block.segments[0].analogsignals[channel_1_idx].sampling_period.magnitude

      
        # .magnitude and .sampling_period are "property"/ a built-in function inside neo
        # In order to find out what they really are
        # print(inspect.getsource(type(sig).sampling_period.fget))
        # sp = sig.sampling_period
        # print(inspect.getsource(type(sp).magnitude.fget))
        # this output two function definition as follows:
        # def sampling_period(self):return 1.0 / self.sampling_rate
        # def magnitude(self): return self.view(type=np.ndarray)


      
        self.time_units = str(neo_block.segments[0].t_start).split(" ")[1]
        # t_start is the offset of the current trace, can be found by pprint(sig.__dict__)
        # '_t_start': array(0.021) * s,
        # so this mean to split the 0.021 and s by empty space, to become '0.021', 's'
        # then [1] keep the unit label
        # so my offset is in seconds, not in ms (the Easy electrophysiology software gui printed ms, which was likely wrong)


        self.channel_1_data = np.zeros((self.num_recs, self.num_samples), dtype=np.float64)
        self.channel_2_data = np.zeros((self.num_recs, self.num_samples), dtype=np.float64)
        self.time_array = np.zeros((self.num_recs, self.num_samples), dtype=np.float64)

        # np.zeros(shape, dtype) creates a new numpy array filled with zeros
        # we allocate an empty array before filling it in with real data 
        # the shape is (self.num_recs, self.num_samples) means (rows, columns)
        # time_array is the corresponding time for each sample when it was collected
      
        self.num_data_channels = num_chans


        self.time_offset = False
        self.channel_1_type = channel_1_type
        self.channel_2_type = channel_2_type
        self.channel_1_idx = channel_1_idx
        self.channel_2_idx = channel_2_idx
        self.channel_1_units = channel_1_units
        self.channel_2_units = channel_2_units
        self.tags = ""
        self.all_channels = channels  

        self.norm_first_deriv_data: Optional[NpArray64] = None
        self.recording_type: Literal["current_clamp", "voltage_clamp", "unknown"]
        self.t_start: float  # start time (first record)
        self.t_stop: float  # end time (final record)
        self.records_are_contiguous: bool  # whether time is contiguous between records
        self.min_max_time: NpArray64
        self.rec_time: NpArray64

        # Because we did not define "load_settings" by self.load_settings as in GitHub
        # I removed this block "if self.load_setting is not None:self.fix_channels_to_current_or_voltage_clamp()"


        if self.channel_1_type == "Vm" and (self.channel_2_idx is None or self.channel_2_type == "Im"):
            self.recording_type = "current_clamp"
        elif self.channel_1_type == "Im" and (self.channel_2_idx is None or self.channel_2_type == "Vm"):
            self.recording_type = "voltage_clamp"
        else:
            self.recording_type = "unknown"

        # Because file extension is never .DAT (always .abf)
        # I removed the block "if file_ext.upper() == ".DAT":........)

        if self.channel_1_type == "Im":
            self.channel_1_data, final_units = self.extract_data_from_array(
                self.channel_1_idx, neo_block, self.channel_1_type, self.channel_1_units
            )
            self.channel_1_units = final_units

            if self.channel_2_idx is not None:
                self.channel_2_data, final_units = self.extract_data_from_array(
                    self.channel_2_idx, neo_block, self.channel_2_type, self.channel_2_units
                )
                self.channel_2_units = final_units

        elif self.channel_1_type == "Vm":
            self.channel_1_data, final_units = self.extract_data_from_array(
                self.channel_1_idx, neo_block, self.channel_1_type, self.channel_1_units
            )
            self.channel_1_units = final_units

            if self.channel_2_idx is not None:
                self.channel_2_data, final_units = self.extract_data_from_array(
                    self.channel_2_idx, neo_block, self.channel_2_type, self.channel_2_units
                )
                self.channel_2_units = final_units

                # I deleted the "elif file_ext.upper() == ".ABF":self.handle_generate_axon_protocol(reader, cfgs)"
                # Because we don't have "cfgs", no GUI,
                # Although our file extension is abf, the code will never run to this evaluation, because our "self.channel_1_type == "Vm":" is never true
                # so it was save to remove this block
        self.time_array, self.t_start, self.t_stop = self.extract_time_array(neo_block)
        self.check_and_clean_data()

        # file format specified fields.
        if file_ext.upper() == ".ABF":
            self.update_tags(reader)

        return

    # I think this function was never called since we dont have load settings
    def fix_channels_to_current_or_voltage_clamp(self):
        if self.load_setting == "current_clamp":
            self.channel_1_type = "Vm"
            self.channel_1_units = "mV"
            if self.channel_2_type is not None:
                self.channel_2_type = "Im"
                self.channel_2_units = "pA"

        elif self.load_setting == "voltage_clamp":
            self.channel_1_type = "Im"
            self.channel_1_units = "pA"
            if self.channel_2_type is not None:
                self.channel_2_type = "Vm"
                self.channel_2_units = "mV"
        else:
            assert self.load_setting is None

    def get_primary_data(self):
        return self.channel_1_data

    def get_secondary_data(self):
        return self.channel_2_data

    def set_data_by_recording_type(self,primary_channel_array,secondary_channel_array):
        """
        Sets data based on analysis type (see get_primary_data()).
        The passed data should match the shape of the existing data
        Must call `set_data_params` and update the GUI after this function.
        """
        assert self.recording_type in ["current_clamp", "voltage_clamp", "unknown"]
        assert self.channel_1_type in ["Im", "Vm"]
        if self.channel_2_type is not None:
            assert self.channel_2_type in ["Im", "Vm"]

        if primary_channel_array is not None:
            self.channel_1_data = primary_channel_array

        if secondary_channel_array is not None:
            self.channel_2_data = secondary_channel_array

    def get_norm_first_deriv_data(self) -> NpArray64:
        """ """
        sample_spacing_in_ms = self.ts * 1000

        if self.norm_first_deriv_data is None:
            self.norm_first_deriv_data = np.diff(self.get_primary_data(), append=0) / sample_spacing_in_ms
            self.norm_first_deriv_data.setflags(write=False)

        return self.norm_first_deriv_data

    def check_and_clean_data(self) -> None:

        assert self.t_start is not None and self.t_stop is not None, "Type Check t_start and t_stop"

        if self.time_units == "ms":
            self.t_start /= 1000
            self.t_stop /= 1000
            self.time_array /= 1000
            self.time_units = "s"

        if self.t_start != 0:
            self.time_offset = np.array(self.t_start)


    def extract_time_array(self, neo_block: Block) -> Tuple[NpArray64, float, float]:
        """ """
        channel_time_check = True

        if self.num_recs == 1:
            time_array = core_analysis_methods.generate_time_array(
                neo_block.segments[0].analogsignals[self.channel_1_idx].t_start.magnitude,
                neo_block.segments[0].analogsignals[self.channel_1_idx].t_stop.magnitude,
                self.num_samples,
                self.ts,
            )[np.newaxis, :]

        else:

            time_array = np.zeros((self.num_recs, self.num_samples), dtype=np.float64)

            for rec in range(self.num_recs):
                time_ = core_analysis_methods.generate_time_array(
                    neo_block.segments[rec].analogsignals[self.channel_1_idx].t_start.magnitude,
                    neo_block.segments[rec].analogsignals[self.channel_1_idx].t_stop.magnitude,
                    self.num_samples,
                    self.ts,
                )
                time_array[rec, :] = time_.squeeze()

                if not self.channel_times_are_equal(neo_block, rec):
                    channel_time_check = False


        t_start = float(neo_block.segments[0].analogsignals[self.channel_1_idx].t_start)
        t_stop = float(neo_block.segments[self.num_recs - 1].analogsignals[self.channel_1_idx].t_stop)

        # I deleted the following block because we dont need to print error messgae
        # if not channel_time_check:utils.show_messagebox("Time Error","The timing of channel 1 and channel 2 are not ""the same. The time units for channel 2 will ""be incorrect. ""Please contact support@easyelectrophysiology.com",)
        return time_array, t_start, t_stop

    def channel_times_are_equal(self, neo_block: Block, rec: Int) -> bool:

        if self.channel_2_idx is not None and len(neo_block.segments[rec].analogsignals) == 2:
            if (
                neo_block.segments[rec].analogsignals[self.channel_1_idx].t_start.magnitude
                != neo_block.segments[rec].analogsignals[self.channel_2_idx].t_start.magnitude
                or neo_block.segments[rec].analogsignals[self.channel_1_idx].t_stop.magnitude
                != neo_block.segments[rec].analogsignals[self.channel_2_idx].t_stop.magnitude
            ):
                return False
        return True


    def extract_data_from_array(
        self, data_idx: Int, neo_block: Block, channel_type, channel_units
    ) -> Tuple[NpArray64, str]:
        
        if self.num_recs == 1:
            array = neo_block.segments[0].analogsignals[data_idx].magnitude.T.astype(np.float64)
            assert array.shape[0] == 1
        else:
            array = np.zeros((self.num_recs, self.num_samples), dtype=np.float64)

            for rec in range(self.num_recs):
                data = neo_block.segments[rec].analogsignals[data_idx].magnitude
                array[rec, :] = data.squeeze()

        final_units = self.scale_array_in_place_if_required(channel_type, channel_units, array)

        return array, final_units

    # I dont think we ever call this function, might be safe to delete
    def load_heka_metadata(self, channel_1_type, channel_1_units, reader):
        """ """
        first_channel_offsets = np.array(reader.header["heka_metadata"]["zero_offsets"][self.channel_1_idx])

        if first_channel_offsets is not None:
            self.scale_array_in_place_if_required(channel_1_type, channel_1_units, array=first_channel_offsets)

        heka_metadata = {
            "primary_channel_zero_offsets": first_channel_offsets,
            "add_zero_offset": reader.header["heka_metadata"]["add_zero_offset"],
        }
        return heka_metadata

    def scale_array_in_place_if_required(self, channel_type, channel_units, array):
        """ """
        assert channel_type in ["Im", "Vm"], "`channel_type` must be `Im` or `Vm`."


        if channel_type == "Im":
            if channel_units != "pA":
                conversion_to_pa_table = core_analysis_methods.get_conversion_to_pa_table()
                array *= conversion_to_pa_table[channel_units]
            final_units = "pA"

        elif channel_type == "Vm":
            if channel_units != "mV":
                conversion_to_mv_table = core_analysis_methods.get_conversion_to_mv_table()
                array *= conversion_to_mv_table[channel_units]
            final_units = "mV"

        return final_units


    ##########################################
    #### DO the GITHUB CODE HAS UTILS???? ####
    ##########################################
    def extract_axon_protocol(self, reader: AxonIO) -> Tuple[NpArray64, str]:

        array = utils.np_empty_nan((self.num_recs, self.num_samples))
        protocol = reader.read_protocol()
        units = protocol[0].analogsignals[0].units.dimensionality.string
        for rec in range(self.num_recs):
            array[rec] = np.squeeze(protocol[rec].analogsignals[0].magnitude)

        return array, units


    # Deleted the funtcion "handle_generate_axon_protocol()"
    # because we dont have cfgs, and no need error message
    

    def update_tags(self, reader: AxonIO) -> None:
        """
        Update self.tags with tags (Axon Instruments feature only)
        """
        all_tags = ""
        for i in range(len(reader._axon_info["listTag"])):
            try:
                tag = reader._axon_info["listTag"][i]["sComment"].decode("utf-8").strip() + " "
            except UnicodeDecodeError:
                tag = "Could not load tags.\n" "Please contact support@easyelectrophysiology.com " "for more details."
            if i > 0:
                tag = "Tag {0}: ".format(str(i + 1)) + tag
            all_tags += tag
        self.tags = all_tags


    # deleted "Multi-file importdata checks" section with
    # "check_loaded_files_match()"

