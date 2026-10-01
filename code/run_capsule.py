""" Writes RAW ephys and LFP to an NWB file """

import sys
import warnings
import argparse
from pathlib import Path
import numpy as np
import os
import json
import time
import logging
import datetime as dt
from datetime import datetime
from uuid import uuid4

import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import probeinterface as pi

from neo.rawio import OpenEphysBinaryRawIO

from neuroconv.tools.nwb_helpers import (
    configure_backend,
    get_default_backend_configuration,
)
from neuroconv.tools.spikeinterface.spikeinterface import (
    add_recording_to_nwbfile,
    add_recording_metadata_to_nwbfile
)

from pynwb import NWBHDF5IO, NWBFile
from pynwb.file import Device
from hdmf_zarr import NWBZarrIO

# for NWB Zarr, let's use built-in compressors, so thay can be read without Python
from numcodecs import Blosc

from aind_nwb_utils.utils import get_ephys_devices_from_metadata

warnings.filterwarnings("ignore")


# filter and resample LFP
lfp_filter_kwargs = dict(freq_min=0.5, freq_max=500, ignore_low_freq_error=True)
lfp_sampling_rate = 2500
lfp_save_chunk_duration = "60s"

# default compressors
default_electrical_series_compressors = dict(hdf5="gzip", zarr=Blosc(cname="zstd", clevel=9, shuffle=Blosc.BITSHUFFLE))

# default event line from open ephys
data_folder = Path("../data/")
scratch_folder = Path("../scratch/")
results_folder = Path("../results/")

parser = argparse.ArgumentParser(description="Export Ecephys data to NWB")
# positional arguments
backend_group = parser.add_mutually_exclusive_group()
backend_help = "NWB backend. It can be either 'hdf5' or 'zarr'."
backend_group.add_argument(
    "--backend", choices=["hdf5", "zarr"], default="zarr", help=backend_help
)
backend_group.add_argument("static_backend", nargs="?", help=backend_help)


stub_group = parser.add_mutually_exclusive_group()
stub_help = "Write a stub version for testing"
stub_group.add_argument("--stub", action="store_true", help=stub_help)
stub_group.add_argument("static_stub", nargs="?", default="false", help=stub_help)

stub_seconds_group = parser.add_mutually_exclusive_group()
stub_seconds_help = "Duration of stub recording"
stub_seconds_group.add_argument("--stub-seconds", default=10, help=stub_seconds_help)
stub_seconds_group.add_argument("static_stub_seconds", nargs="?", default="10", help=stub_seconds_help)

write_lfp_group = parser.add_mutually_exclusive_group()
write_lfp_help = "Whether to write LFP electrical series"
write_lfp_group.add_argument("--skip-lfp", action="store_true", help=write_lfp_help)
write_lfp_group.add_argument("static_write_lfp", nargs="?", default="true", help=write_lfp_help)

write_raw_group = parser.add_mutually_exclusive_group()
write_raw_help = "Whether to write RAW electrical series"
write_raw_group.add_argument("--write-raw", action="store_true", help=write_raw_help)
write_raw_group.add_argument("static_write_raw", nargs="?", default="false", help=write_raw_help)

lfp_temporal_subsampling_group = parser.add_mutually_exclusive_group()
lfp_temporal_subsampling_help = (
    "Ratio of input samples to output samples in time. Use 0 or 1 to keep all samples. Default is 2."
)
lfp_temporal_subsampling_group.add_argument("--lfp_temporal_factor", default=2, help=lfp_temporal_subsampling_help)
lfp_temporal_subsampling_group.add_argument("static_lfp_temporal_factor", nargs="?", help=lfp_temporal_subsampling_help)

lfp_spatial_subsampling_group = parser.add_mutually_exclusive_group()
lfp_spatial_subsampling_help = (
    "Controls number of channels to skip in spatial subsampling. Use 0 or 1 to keep all channels. Default is 4."
)
lfp_spatial_subsampling_group.add_argument("--lfp_spatial_factor", default=4, help=lfp_spatial_subsampling_help)
lfp_spatial_subsampling_group.add_argument("static_lfp_spatial_factor", nargs="?", help=lfp_spatial_subsampling_help)

lfp_highpass_filter_group = parser.add_mutually_exclusive_group()
lfp_highpass_filter_help = (
    "Cutoff frequency for highpass filter to apply to the LFP recorsings. Default is 0.1. (Use 0 to skip)."
)
lfp_highpass_filter_group.add_argument("--lfp_highpass_freq_min", default=0.1, help=lfp_highpass_filter_help)
lfp_highpass_filter_group.add_argument("static_lfp_highpass_freq_min", nargs="?", help=lfp_highpass_filter_help)

# common median referencing for probes in agar
lfp_surface_channel_agar_group = parser.add_mutually_exclusive_group()
lfp_surface_channel_help = "Index of surface channel (e.g. index 0 corresponds to channel 1) of probe for common median referencing for probes in agar. Pass in as JSON string where key is probe and value is surface channel (e.g. \"{'ProbeA': 350, 'ProbeB': 360}\")"
lfp_surface_channel_agar_group.add_argument(
    "--surface_channel_agar_probes_indices", help=lfp_surface_channel_help, default="", type=str
)
lfp_surface_channel_agar_group.add_argument(
    "static_surface_channel_agar_probes_indices", help=lfp_surface_channel_help, nargs="?", type=str
)

parser.add_argument("--params", default=None, help="Path to the parameters file or JSON string. If given, it will override all other arguments.")


def run() -> None:
    """Entrypoint for the NWB packaging ecephys capsule."""
    t_export_start = time.perf_counter()

    args = parser.parse_args()

    PARAMS = args.params

    LOGGING = None

    if PARAMS is not None:
        try:
            # try to parse the JSON string first to avoid file name too long error
            nwb_ecephys_params = json.loads(PARAMS)
        except json.JSONDecodeError:
            if Path(PARAMS).is_file():
                with open(PARAMS, "r") as f:
                    nwb_ecephys_params = json.load(f)
            else:
                raise ValueError(f"Invalid parameters: {PARAMS} is not a valid JSON string or file path")
        NWB_BACKEND = nwb_ecephys_params.get("backend", "zarr")
        STUB_TEST = nwb_ecephys_params.get("stub", False)
        STUB_SECONDS = float(nwb_ecephys_params.get("stub_seconds", 10))
        WRITE_LFP = nwb_ecephys_params.get("write_lfp", True)
        WRITE_RAW = nwb_ecephys_params.get("write_raw", False)
        TEMPORAL_SUBSAMPLING_FACTOR = int(nwb_ecephys_params.get("lfp_temporal_factor", 2))
        SPATIAL_CHANNEL_SUBSAMPLING_FACTOR = int(nwb_ecephys_params.get("lfp_spatial_factor", 4))
        HIGHPASS_FILTER_FREQ_MIN = float(nwb_ecephys_params.get("lfp_highpass_freq_min", 0.1))
        SURFACE_CHANNEL_AGAR_PROBES_INDICES = nwb_ecephys_params.get("surface_channel_agar_probes_indices", None)
    else:
        with open("params.json", "r") as f:
            nwb_ecephys_params = json.load(f)

        NWB_BACKEND = args.static_backend or args.backend
        stub = args.stub or args.static_stub
        if args.stub:
            STUB_TEST = True
        else:
            STUB_TEST = True if args.static_stub == "true" else False
        STUB_SECONDS = float(args.stub_seconds) or float(args.static_stub_secods)

        if args.skip_lfp:
            WRITE_LFP = False
        else:
            WRITE_LFP = True if args.static_write_lfp == "true" else False

        if args.write_raw:
            WRITE_RAW = True
        else:
            WRITE_RAW = True if args.static_write_raw == "true" else False

        TEMPORAL_SUBSAMPLING_FACTOR = args.static_lfp_temporal_factor or args.lfp_temporal_factor
        TEMPORAL_SUBSAMPLING_FACTOR = int(TEMPORAL_SUBSAMPLING_FACTOR)
        SPATIAL_CHANNEL_SUBSAMPLING_FACTOR = args.static_lfp_spatial_factor or args.lfp_spatial_factor
        SPATIAL_CHANNEL_SUBSAMPLING_FACTOR = int(SPATIAL_CHANNEL_SUBSAMPLING_FACTOR)
        HIGHPASS_FILTER_FREQ_MIN = args.static_lfp_highpass_freq_min or args.lfp_highpass_freq_min
        HIGHPASS_FILTER_FREQ_MIN = float(HIGHPASS_FILTER_FREQ_MIN)
        SURFACE_CHANNEL_AGAR_PROBES_INDICES = (
            args.static_surface_channel_agar_probes_indices or args.surface_channel_agar_probes_indices
        )
        if SURFACE_CHANNEL_AGAR_PROBES_INDICES != "":
            SURFACE_CHANNEL_AGAR_PROBES_INDICES = json.loads(SURFACE_CHANNEL_AGAR_PROBES_INDICES)
        else:
            SURFACE_CHANNEL_AGAR_PROBES_INDICES = None

    # TODO: temporary - remove from params.json when logging is distributed by pipeline
    LOGGING = nwb_ecephys_params.pop("logging", None)

    # Use CO_CPUS/N_JOBS_EXT env variable if available
    N_JOBS_EXT = os.getenv("CO_CPUS") or os.getenv("N_JOBS_EXT")
    N_JOBS = int(N_JOBS_EXT) if N_JOBS_EXT is not None else -1
    job_kwargs = dict(n_jobs=N_JOBS, progress_bar=False, mp_context="spawn")
    si.set_global_job_kwargs(**job_kwargs)

    # setup logging before any other logging call
    if LOGGING is None:
        logging.basicConfig(level="INFO", stream=sys.stdout, format="%(message)s")
    else:
        if LOGGING["package"] == "logging":
            logging_cfg = LOGGING.get("logging_cfg", {})
            logging.basicConfig(stream=sys.stdout, **logging_cfg)
        elif LOGGING["package"] == "log-schema":
            import log_schema

            pipeline_name = LOGGING.get("pipeline_name", "AIND Ephys Pipeline")
            acquisition_name = LOGGING.get("acquisition_name", None)

            if acquisition_name is None:
                data_description_json = list(data_folder.glob("**/data_description.json"))
                if len(data_description_json) > 0:
                    data_description_json = data_description_json[0]
                    with open(data_description_json, "r") as f:
                        data_description = json.load(f)
                    acquisition_name = data_description["name"]

            config = LOGGING.get("logging_cfg")
            if config is not None and len(config) == 0:
                config = None
            log_schema.setup_logging(
                config=config,
                model={
                    "pipeline_name": pipeline_name,
                    "acquisition_name": acquisition_name,
                    "process_name": "NWB Packaging Ecephys"
                }
            )

    logging.info("Begin processing...", extra={"event_type": "stage_start"})
    logging.info("\n\nNWB EXPORT ECEPHYS")

    logging.info(f"Running NWB conversion with the following parameters:")
    logging.info(f"Stub test: {STUB_TEST}")
    logging.info(f"Stub seconds: {STUB_SECONDS}")
    logging.info(f"Write LFP: {WRITE_LFP}")
    logging.info(f"Write RAW: {WRITE_RAW}")
    logging.info(f"Temporal subsampling factor: {TEMPORAL_SUBSAMPLING_FACTOR}")
    logging.info(f"Spatial subsampling factor: {SPATIAL_CHANNEL_SUBSAMPLING_FACTOR}")
    logging.info(f"Highpass filter frequency: {HIGHPASS_FILTER_FREQ_MIN}")
    logging.info(f"Surface channel indices for agar probes: {SURFACE_CHANNEL_AGAR_PROBES_INDICES}")

    # find base NWB file
    nwb_files = [p for p in data_folder.iterdir() if p.name.endswith(".nwb") or p.name.endswith(".nwb.zarr")]
    nwbfile_input_path = None
    if len(nwb_files) == 1:
        nwbfile_input_path = nwb_files[0]

    if nwbfile_input_path is not None:
        logging.info(f"Found NWB file: {nwbfile_input_path}. Setting up NWB backend based on input file type.")
        if nwbfile_input_path.is_dir():
            assert (nwbfile_input_path / ".zattrs").is_file(), f"{nwbfile_input_path.name} is not a valid Zarr folder"
            NWB_BACKEND = "zarr"
        else:
            NWB_BACKEND = "hdf5"

    logging.info(f"NWB backend: {NWB_BACKEND}")
    if NWB_BACKEND == "zarr":
        io_class = NWBZarrIO
    else:
        io_class = NWBHDF5IO

    job_json_files = [p for p in data_folder.glob('**/*.json') if "job" in p.name]
    job_dicts = []
    for job_json_file in job_json_files:
        with open(job_json_file) as f:
            job_dict = json.load(f)
        job_dicts.append(job_dict)
    logging.info(f"Found {len(job_dicts)} JSON job files")

    # check for timestamps to overwrite recording timestamps
    timestamps_folder = data_folder / "timestamps"

    # we create a result NWB file for each experiment/recording
    session_names = np.unique([job_dict["session_name"] for job_dict in job_dicts])

    for session_name in session_names:
        logging.info(f"Session: {session_name}")
        # filter job_dicts for this session
        job_dicts_session = [jd for jd in job_dicts if jd["session_name"] == session_name]
        input_folder = job_dicts_session[0].get("input_folder")

        recording_names = [job_dict["recording_name"] for job_dict in job_dicts_session]

        # find blocks and recordings
        block_ids = []
        recording_ids = []
        stream_names = []
        for recording_name in recording_names:
            if "group" in recording_name:
                block_str = recording_name.split("_")[0]
                recording_str = recording_name.split("_")[-2]
                stream_name = "_".join(recording_name.split("_")[1:-2])
            else:
                block_str = recording_name.split("_")[0]
                recording_str = recording_name.split("_")[-1]
                stream_name = "_".join(recording_name.split("_")[1:-1])

            if block_str not in block_ids:
                block_ids.append(block_str)
            if recording_str not in recording_ids:
                recording_ids.append(recording_str)
            if stream_name not in stream_names:
                stream_names.append(stream_name)
        # note: in case of groups, we will need to aggregate the data for each stream into a single recording
        streams_to_process = []
        for stream_name in stream_names:
            # Skip NI-DAQ
            if "NI-DAQ" in stream_name:
                continue
            # LFP are handled later
            if "LFP" in stream_name:
                continue
            streams_to_process.append(stream_name)

        block_ids = sorted(block_ids)
        recording_ids = sorted(recording_ids)
        streams_to_process = sorted(streams_to_process)

        logging.info(f"Number of NWB files to write: {len(block_ids) * len(recording_ids)}")

        logging.info(f"Number of streams to write for each file: {len(streams_to_process)}")

        # Construct 1 nwb file per experiment - streams are concatenated!
        nwb_output_files = []
        electrical_series_to_configure = []
        nwb_output_files = []
        for block_index, block_str in enumerate(block_ids):
            for segment_index, recording_str in enumerate(recording_ids):
                # add recording/experiment id if needed
                nwbfile = None
                read_io = None
                if nwbfile_input_path is not None:
                    nwb_original_file_name = nwbfile_input_path.stem
                    if block_str in nwb_original_file_name and recording_str in nwb_original_file_name:
                        nwb_file_name = nwb_original_file_name
                    else:
                        nwb_file_name = f"{nwb_original_file_name}_{block_str}_{recording_str}"
                    read_io = io_class(str(nwbfile_input_path), "r")
                    logging.info(f"Using existing NWB file: {nwb_file_name}")
                    nwbfile = read_io.read()
                else:
                    # if no input file, create a new one
                    nwb_file_name = f"{session_name}_{block_str}_{recording_str}"

                        
                    if input_folder is not None:
                        try:
                            from aind_nwb_utils.utils import create_base_nwb_file
                            nwbfile = create_base_nwb_file(Path(input_folder))
                        except:
                            logging.info(f"Failed to create base NWB file from metadata.")

                    if nwbfile is None:
                        from pynwb.testing.mock.file import mock_Subject
                        logging.info(f"Creating NWB file with info.")
                        
                        subject = mock_Subject()
                        timezone_info = datetime.now(dt.timezone.utc).astimezone().tzinfo
                        session_start_date_time = datetime.now().replace(
                            tzinfo=timezone_info
                        )
                        institution = None
                        session_id = session_name
                        asset_name = session_id

                        # Store and write NWB file
                        nwbfile = NWBFile(
                            session_description="NWB file generated by AIND pipeline",
                            identifier=str(uuid4()),
                            session_start_time=session_start_date_time,
                            institution=institution,
                            subject=subject,
                            session_id=session_id,
                        )

                # add suffix for stub test
                if STUB_TEST:
                    nwb_file_name = f"{nwb_file_name}_stub"

                nwbfile_output_path = results_folder / f"{nwb_file_name}.nwb"

                # Find probe devices (this will only work for AIND)
                devices_from_metadata, target_locations = None, None
                add_probe_device_from_rig = False
                if input_folder is not None:
                    devices_from_metadata, target_locations = get_ephys_devices_from_metadata(
                        input_folder
                    )

                probe_device_names = []
                for stream_index, stream_name in enumerate(streams_to_process):
                    recording_name = f"{block_str}_{stream_name}_{recording_str}"
                    logging.info(f"Processing {recording_name}")

                    # load JSON and recordings
                    # we need lists because multiple groups are saved to different JSON files
                    recording_job_dicts = []
                    for job_dict in job_dicts_session:
                        if recording_name in job_dict["recording_name"]:
                            recording_job_dicts.append(job_dict)

                    recording_lfp = None
                    recordings = []
                    recordings_lfp = []
                    logging.info(f"\tLoading {recording_name} from {len(recording_job_dicts)} JSON files")
                    if len(recording_job_dicts) > 1:
                        # in case of multiple groups, sort by group names
                        sort_idxs = np.argsort([jd["recording_name"] for jd in recording_job_dicts])
                        recording_job_dicts_sorted = np.array(recording_job_dicts)[sort_idxs]
                    else:
                        recording_job_dicts_sorted = recording_job_dicts
                    for recording_job_dict in recording_job_dicts_sorted:
                        recording = si.load(recording_job_dict["recording_dict"], base_folder=data_folder)
                        recording_name = recording_job_dict["recording_name"]
                        skip_times = recording_job_dict.get("skip_times", False)
                        if skip_times:
                            recording.reset_times()
                        if recording.get_dtype().kind == "u":
                            logging.info(
                                f"Recording has unsigned integer dtype {recording.get_dtype()}. "
                                "Converting to signed integer."
                            )
                            recording = spre.unsigned_to_signed(recording)
                        timestamps_file = timestamps_folder / f"{recording_name}.npy"
                        if timestamps_file.is_file():
                            logging.info(f"\tSetting synced timestamps from {timestamps_file}")
                            timestamps = np.load(timestamps_file)
                            recording.set_times(timestamps, with_warning=False)
                        recordings.append(recording)

                        logging.info(f"\t\t{recording_job_dict['recording_name']}")
                        if "recording_lfp_dict" in recording_job_dict:
                            logging.info(f"\tLoading associated LFP recording")
                            recording_lfp = si.load(recording_job_dict["recording_lfp_dict"], base_folder=data_folder)
                            if skip_times:
                                recording_lfp.reset_times()
                            if recording_lfp.get_dtype().kind == "u":
                                logging.info(
                                    f"Recording LFP has unsigned integer dtype {recording_lfp.get_dtype()}. "
                                    "Converting to signed integer."
                                )
                                recording_lfp = spre.unsigned_to_signed(recording_lfp)
                            timestamps_file_lfp = timestamps_folder / f"{recording_name}_lfp.npy"
                            if timestamps_file_lfp.is_file():
                                logging.info(f"\tSetting synced LFP timestamps from {timestamps_file_lfp}")
                                timestamps_lfp = np.load(timestamps_file_lfp)
                                recording_lfp.set_times(timestamps_lfp, with_warning=False)
                            recordings_lfp.append(recording_lfp)
                            logging.info(f"\t\t{recording_lfp}")

                    # for multiple groups, aggregate channels
                    if len(recording_job_dicts_sorted) > 1:
                        logging.info(f"\t\tAggregating channels from {len(recordings)} groups")
                        recording = si.aggregate_channels(recordings)
                        # probes_info get lost in aggregation, so we need to manually set them
                        recording.annotate(
                            probes_info=recordings[0].get_annotation("probes_info")
                        )
                        # remove aggregation key property, since it causes typing issue in NWB export
                        if "aggregation_key" in recording.get_property_keys():
                            recording.delete_property("aggregation_key")
                        if len(recordings_lfp) > 0:
                            recording_lfp = si.aggregate_channels(recordings_lfp)
                            recording_lfp.annotate(
                                probes_info=recordings_lfp[0].get_annotation("probes_info")
                            )
                            if "aggregation_key" in recording_lfp.get_property_keys():
                                recording_lfp.delete_property("aggregation_key")

                    if STUB_TEST:
                        end_frame = int(STUB_SECONDS * recording.sampling_frequency)
                        recording = recording.frame_slice(start_frame=0, end_frame=end_frame)
                        if recording_lfp is not None:
                            end_frame = int(STUB_SECONDS * recording_lfp.sampling_frequency)
                            recording_lfp = recording_lfp.frame_slice(start_frame=0, end_frame=end_frame)

                    # Add device and electrode group
                    # For the NWB case, since the parser only read channel locations, the job-dispatch creates
                    # a probe with the correct probe_device_name, so that neuroconv uses the right existing device
                    if recording_job_dicts[0].get("probe_dict") is not None:
                        logging.info(f"\tAdding probe information from job-dispatch metadata")
                        probe_dict = recording_job_dicts[0]["probe_dict"]
                        probe = pi.Probe.from_dict(probe_dict)
                        electrode_group_location = probe.annotations.get("electrode_group_location", "unknown")
                    else:
                        logging.info(f"\tAdding probe information from recording metadata")
                        probegroup = recording.get_probegroup()
                        assert len(probegroup.probes) == 1, (
                            "Grouping failed for this session. Each stream should be associated with a single probe!"
                        )
                        probe = probegroup.probes[0]
                        electrode_group_location = "unknown"

                    # 1. Look for AIND devices in metadata and use them if they match the stream name
                    probe_device_name = None
                    if devices_from_metadata:
                        for device_name, device in devices_from_metadata.items():
                            # find probe device name associated to stream
                            probe_no_spaces = device_name.replace(" ", "")
                            if probe_no_spaces in stream_name:
                                probe_device_name = device_name
                                electrode_group_location = target_locations.get(device_name, "unknown")
                                logging.info(
                                    f"\tFound device from metadata: {probe_device_name} at location {electrode_group_location}"
                                )
                                # 1a. Apply fix for Quad Base probes to get probe device name from probe metadata instead of rig.json,
                                # since rig.json has the same name for all shanks but we need to differentiate them
                                model_name = probe.model_name
                                model_description = probe.description
                                if (model_name is not None and "Quad Base" in model_name) or \
                                    (model_description is not None and "Quad Base" in model_description):
                                    logging.info(f"Detected Quad Base: changing name from {probe_device_name} to {probe.name}")
                                    probe_device_name = probe.name
                                break

                    # 2. If no metadata devices, use probeinterface probes info from recording annotations
                    if probe_device_name is None:
                        probe_device_name = probe.name or probe.model_name or "Probe"
                        logging.info(f"\tAdding probe information from recording metadata")

                    # 3. Add probe to NWB
                    probe_device_manufacturer = probe.manufacturer
                    probe_model_name = probe.model_name
                    probe_serial_number = probe.serial_number
                    probe_description = probe.description
                    probe_device_description = ""

                    if probe_model_name is not None:
                        probe_device_description += f"Model: {probe_model_name}"
                    if probe_serial_number is not None:
                        if len(probe_device_description) > 0:
                            probe_device_description += " - "
                        probe_device_description += f"Serial number: {probe_serial_number}"
                    if probe_description is not None:
                        if len(probe_device_description) > 0:
                            probe_device_description += " - "
                        probe_device_description += f"Description: {probe_description}"
                    # this is needed to account for a case where multiple streams have the same device name
                    if len(streams_to_process) > 1 and probe_device_name in probe_device_names:
                        probe_device_name = f"{probe_device_name}-{stream_index}"
                    probe_device = Device(
                        name=probe_device_name,
                        description=probe_device_description,
                        manufacturer=probe_device_manufacturer,
                    )
                    if probe_device_name not in nwbfile.devices:
                        nwbfile.add_device(probe_device)
                        logging.info(f"\tAdded probe device: {probe_device.name} - {probe_device.description}")
                    probe_device_names.append(probe_device_name)

                    # Add electrode metadata
                    electrode_metadata = dict(
                        Ecephys=dict(
                            Device=[dict(name=probe_device_name)],
                        )
                    )
                    # Add channel properties (group_name property to associate electrodes with group)
                    channel_groups = recording.get_channel_groups()
                    if len(np.unique(channel_groups)) == 1:
                        recording.set_channel_groups([probe_device_name] * recording.get_num_channels())
                        electrode_groups_metadata = [
                            dict(
                                name=probe_device_name,
                                description=f"Recorded electrodes from probe {probe_device_name}",
                                location=electrode_group_location,
                                device=probe_device_name,
                            )
                        ]
                    else:
                        recording.set_channel_groups([f"{probe_device_name}_group{g}" for g in channel_groups])
                        channel_groups_unique = np.unique(recording.get_channel_groups())
                        electrode_groups_metadata = [
                            dict(
                                name=group,
                                description=f"Recorded electrodes from group {group}",
                                location=electrode_group_location,
                                device=probe_device_name,
                            )
                            for group in channel_groups_unique
                        ]
                    electrode_metadata["Ecephys"]["ElectrodeGroup"] = electrode_groups_metadata

                    if WRITE_RAW:
                        electrical_series_name = f"ElectricalSeries{probe_device_name}"
                        electrical_series_metadata = {
                            electrical_series_name: dict(
                                name=f"ElectricalSeries{probe_device_name}",
                                description=f"Voltage traces from {probe_device_name}",
                            )
                        }
                        electrode_metadata["Ecephys"].update(electrical_series_metadata)
                        add_electrical_series_kwargs = dict(
                            es_key=f"ElectricalSeries{probe_device_name}", write_as="raw"
                        )

                        logging.info(f"\tAdding RAW data for stream {stream_name} - segment {segment_index}")
                        add_recording_to_nwbfile(
                            recording=recording,
                            nwbfile=nwbfile,
                            metadata=electrode_metadata,
                            always_write_timestamps=True,
                            **add_electrical_series_kwargs,
                        )
                        electrical_series_to_configure.append(add_electrical_series_kwargs["es_key"])
                    else:
                        # always add recording electrodes, as they will be used by Units
                        add_recording_metadata_to_nwbfile(recording=recording, nwbfile=nwbfile, metadata=electrode_metadata)

                    if WRITE_LFP:
                        electrical_series_name = f"ElectricalSeries{probe_device_name}-LFP"
                        electrical_series_metadata = {
                            electrical_series_name: dict(
                                name=f"ElectricalSeries{probe_device_name}-LFP",
                                description=f"LFP voltage traces from {probe_device_name}",
                            )
                        }
                        electrode_metadata["Ecephys"].update(electrical_series_metadata)
                        add_electrical_lfp_series_kwargs = dict(
                            es_key=f"ElectricalSeries{probe_device_name}-LFP",
                            write_as="lfp",
                        )

                        if recording_lfp is None:
                            # Wide-band recording: filter and resample LFP
                            logging.info(
                                f"\tAdding LFP data for stream {stream_name} from wide-band signal - segment {segment_index}"
                            )
                            # added conversion here
                            if recording_lfp.get_dtype().kind == "u":
                                logging.info(
                                    f"Recording LFP has unsigned integer dtype {recording_lfp.get_dtype()}. "
                                    "Converting to signed integer."
                                )
                                recording_lfp = spre.unsigned_to_signed(recording_lfp)
                                
                            recording_lfp = spre.bandpass_filter(recording, **lfp_filter_kwargs)
                            recording_lfp = spre.resample(recording_lfp, lfp_sampling_rate)
                            recording_lfp = spre.astype(recording_lfp, dtype="int16")

                            # there is a bug in with sample mismatches for the last chunk if num_samples not divisible by chunk_size
                            # the workaround is to discard the last samples to make it "even"
                            if recording.get_num_segments() == 1:
                                                            # added conversion here
                                if recording_lfp.get_dtype().kind == "u":
                                    logging.info(
                                        f"Recording LFP has unsigned integer dtype {recording_lfp.get_dtype()}. "
                                        "Converting to signed integer."
                                    )
                                    recording_lfp = spre.unsigned_to_signed(recording_lfp)
                                    
                                recording_lfp = recording_lfp.frame_slice(
                                    start_frame=0,
                                    end_frame=int(
                                        recording_lfp.get_num_samples() // lfp_sampling_rate * lfp_sampling_rate
                                    ),
                                )
                            # set times
                            lfp_period = 1.0 / lfp_sampling_rate
                            for sg_idx in range(recording.get_num_segments()):
                                ts_lfp = (
                                    np.arange(recording_lfp.get_num_samples(sg_idx))
                                    / recording_lfp.sampling_frequency
                                    - recording.get_times(sg_idx)[0]
                                    + lfp_period / 2
                                )
                                recording_lfp.set_times(ts_lfp, segment_index=sg_idx, with_warning=False)
                            save_to_binary = True
                        else:
                            logging.info(f"\tAdding LFP data for {stream_name} from LFP stream - segment {segment_index}")
                            save_to_binary = False
                            # In this case, since LFPs are in a separate stream, we have to reset channel groups
                            channel_groups = recording_lfp.get_channel_groups()
                            if len(np.unique(channel_groups)) == 1:
                                recording_lfp.set_channel_groups([probe_device_name] * recording_lfp.get_num_channels())
                            else:
                                recording_lfp.set_channel_groups([f"{probe_device_name}_group{g}" for g in channel_groups])

                        channel_ids = recording_lfp.get_channel_ids()

                        # re-reference only for agar - subtract median of channels out of brain using surface channel index arg
                        # similar processing to allensdk
                        if SURFACE_CHANNEL_AGAR_PROBES_INDICES is not None:
                            if probe_device_name in SURFACE_CHANNEL_AGAR_PROBES_INDICES:
                                logging.info(f"\t\tCommon median referencing for probe {probe_device_name}")
                                surface_channel_index = SURFACE_CHANNEL_AGAR_PROBES_INDICES[probe_device_name]
                                # get indices of channels out of brain including surface channel
                                reference_channel_indices = np.arange(surface_channel_index, len(channel_ids))
                                reference_channel_ids = channel_ids[reference_channel_indices]
                                # common median reference to channels out of brain
                                recording_lfp = spre.common_reference(
                                    recording_lfp,
                                    reference="global",
                                    ref_channel_ids=reference_channel_ids,
                                )
                            else:
                                logging.info(f"Could not find {probe_device_name} in surface channel dictionary")

                        # spatial subsampling from allensdk - keep every nth channel
                        if SPATIAL_CHANNEL_SUBSAMPLING_FACTOR > 1:
                            logging.info(f"\t\tSpatial subsampling factor: {SPATIAL_CHANNEL_SUBSAMPLING_FACTOR}")
                            channel_ids_to_keep = channel_ids[0 : len(channel_ids) : SPATIAL_CHANNEL_SUBSAMPLING_FACTOR]
                            recording_lfp = recording_lfp.select_channels(channel_ids_to_keep)

                        # time subsampling/decimate
                        if TEMPORAL_SUBSAMPLING_FACTOR > 1:
                            logging.info(f"\t\tTemporal subsampling factor: {TEMPORAL_SUBSAMPLING_FACTOR}")
                            recording_lfp_sub = spre.decimate(recording_lfp, TEMPORAL_SUBSAMPLING_FACTOR)
                            for sg_idx in range(recording.get_num_segments()):
                                lfp_times = recording_lfp.get_times(segment_index=sg_idx)
                                recording_lfp_sub.set_times(lfp_times[::TEMPORAL_SUBSAMPLING_FACTOR], segment_index=sg_idx, with_warning=False)
                            recording_lfp = recording_lfp_sub

                        # high pass filter from allensdk
                        if HIGHPASS_FILTER_FREQ_MIN > 0:
                            logging.info(f"\t\tHighpass filter frequency: {HIGHPASS_FILTER_FREQ_MIN}")
                            recording_lfp = spre.highpass_filter(recording_lfp, freq_min=HIGHPASS_FILTER_FREQ_MIN, ignore_low_freq_error=True)

                        # For streams without a separate LFP, save to binary to speed up conversion later
                        if save_to_binary:
                            logging.info(f"\tSaving preprocessed LFP to binary")
                            recording_lfp = recording_lfp.save(
                                folder=scratch_folder / f"{recording_name}-LFP",
                                verbose=False,
                                overwrite=True,
                                chunk_duration=lfp_save_chunk_duration
                            )

                        logging.info(f"\tAdding LFP recording {recording_lfp}")
                        add_recording_to_nwbfile(
                            recording=recording_lfp,
                            nwbfile=nwbfile,
                            metadata=electrode_metadata,
                            always_write_timestamps=True,
                            **add_electrical_lfp_series_kwargs,
                        )
                        electrical_series_to_configure.append(add_electrical_lfp_series_kwargs["es_key"])

                logging.info(f"Added {len(streams_to_process)} streams")
                logging.info(f"Configuring {NWB_BACKEND} backend")
                backend_configuration = get_default_backend_configuration(nwbfile=nwbfile, backend=NWB_BACKEND)
                es_compressor = default_electrical_series_compressors[NWB_BACKEND]

                for key in backend_configuration.dataset_configurations.keys():
                    if any([es_name in key for es_name in electrical_series_to_configure]) and "timestamps" not in key:
                        backend_configuration.dataset_configurations[key].compression_method = es_compressor
                configure_backend(nwbfile=nwbfile, backend_configuration=backend_configuration)

                logging.info(f"Writing NWB file to {nwbfile_output_path}")
                if NWB_BACKEND == "zarr":
                    write_args = {"link_data": False}
                else:
                    write_args = {}

                t_write_start = time.perf_counter()
                if nwbfile_input_path is not None:
                    # if we have an input file, we read it and write it to the output file
                    export_io = io_class(str(nwbfile_output_path), "w")
                    export_io.export(src_io=read_io, nwbfile=nwbfile, write_args=write_args)
                    read_io.close()
                else:
                    # if no input file, we create a new one
                    nwbfile_output_path = results_folder / f"{nwb_file_name}.nwb"
                    # write the nwb file
                    with io_class(str(nwbfile_output_path), "w") as write_io:
                        write_io.write(nwbfile)
                t_write_end = time.perf_counter()
                elapsed_time_write = np.round(t_write_end - t_write_start, 2)
                logging.info(f"Writing time: {elapsed_time_write}s")
                logging.info(f"Done writing {nwbfile_output_path}")
                nwb_output_files.append(nwbfile_output_path)

    t_export_end = time.perf_counter()
    elapsed_time_export = np.round(t_export_end - t_export_start, 2)
    logging.info(f"NWB EXPORT ECEPHYS time: {elapsed_time_export}s")
    logging.info("Pipeline stage completed", extra={"event_type": "stage_complete"})


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        logging.exception("Pipeline stage failed", extra={"event_type": "stage_error"})
        raise
