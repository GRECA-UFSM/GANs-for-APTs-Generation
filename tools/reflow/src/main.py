import argparse
import pathlib
import datetime
import pandas as pd
import nfstream

def load_dapt2020_flows(dapt2020_flows_path: pathlib.Path) -> pd.DataFrame:
    columns = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Protocol',
       'Timestamp', 'Flow Duration', 'Total Fwd Packet', 'Total Bwd packets',
       'Total Length of Fwd Packet', 'Total Length of Bwd Packet',
       'Fwd Packet Length Max', 'Fwd Packet Length Min',
       'Fwd Packet Length Mean', 'Fwd Packet Length Std',
       'Bwd Packet Length Max', 'Bwd Packet Length Min',
       'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s',
       'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
       'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std',
       'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean',
       'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags',
       'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length',
       'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
       'Packet Length Min', 'Packet Length Max', 'Packet Length Mean',
       'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count',
       'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
       'URG Flag Count', 'CWR Flag Count', 'ECE Flag Count', 'Down/Up Ratio',
       'Average Packet Size', 'Fwd Segment Size Avg', 'Bwd Segment Size Avg',
       'Fwd Bytes/Bulk Avg', 'Fwd Packet/Bulk Avg', 'Fwd Bulk Rate Avg',
       'Bwd Bytes/Bulk Avg', 'Bwd Packet/Bulk Avg', 'Bwd Bulk Rate Avg',
       'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets',
       'Subflow Bwd Bytes', 'FWD Init Win Bytes', 'Bwd Init Win Bytes',
       'Fwd Act Data Pkts', 'Fwd Seg Size Min', 'Active Mean', 'Active Std',
       'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max',
       'Idle Min', 'Activity', 'Stage'
    ]
    
    pcap_flows = pd.read_csv(dapt2020_flows_path, header=0, low_memory=False, delimiter=',')
    
    if 'Flow ID' not in pcap_flows.columns:
        pcap_flows = pd.read_csv(dapt2020_flows_path, header=None, low_memory=False)
        pcap_flows.columns = columns
    
    # A Timestamp está no fuso horário local (Tempe, Arizona, Estados Unidos), que é UTC-7
    pcap_flows['Timestamp'] = pd.to_datetime(pcap_flows['Timestamp'], format='%d/%m/%Y %I:%M:%S %p')
    pcap_flows['Timestamp'] = pcap_flows['Timestamp'] + pd.Timedelta(hours=7)
    pcap_flows['Activity'] = pcap_flows['Activity'].replace('BENIGN', 'Normal')
    pcap_flows['Stage'] = pcap_flows['Stage'].replace('BENIGN', 'Benign')
    pcap_flows.sort_values(by='Timestamp', kind='mergesort', inplace=True, ignore_index=True)     

    return pcap_flows

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate flows from DAPT2020 using NFStream and reassigns the stage labels.')
    parser.add_argument('dataset_path', type=str, help='Path to the DAPT2020 dataset')
    parser.add_argument('output_path', type=str, help='Path to the output directory')

    args = parser.parse_args()
    dataset_path = pathlib.Path(args.dataset_path)
    output_path = pathlib.Path(args.output_path)
    
    for pcap_path in dataset_path.joinpath('pcap-data').glob('*.pcap'):
        print(f"Generating flows for {pcap_path.name}...")
        dapt2020_flow_path = dataset_path.joinpath('csv', pcap_path.name + '_Flow.csv')
        
        if not dapt2020_flow_path.exists():
            print(f"Warning: No corresponding CSV file found for {pcap_path.name}. Skipping...")
            continue
            
        dapt2020_flows = load_dapt2020_flows(dapt2020_flow_path)
        dapt2020_map = {}
        
        for i, flow in dapt2020_flows.iterrows():
            fwd_key = (flow['Src IP'], flow['Src Port'], flow['Dst IP'], flow['Dst Port'], flow['Protocol'])
            bwd_key = (flow['Dst IP'], flow['Dst Port'], flow['Src IP'], flow['Src Port'], flow['Protocol'])
            
            if not dapt2020_map.get(fwd_key):
                dapt2020_map[fwd_key] = []
            
            if not dapt2020_map.get(bwd_key):
                dapt2020_map[bwd_key] = []
            
            dapt2020_map[fwd_key].append((flow['Timestamp'], flow['Flow Duration'], flow['Activity'], flow['Stage']))
            dapt2020_map[bwd_key].append((flow['Timestamp'], flow['Flow Duration'], flow['Activity'], flow['Stage']))

        streamer = nfstream.NFStreamer(
            source=str(pcap_path),
            decode_tunnels=True,
            bpf_filter='ip',
            promiscuous_mode=True,
            snapshot_length=1536,
            idle_timeout=120,
            active_timeout=1800,
            accounting_mode=0,
            udps=None,
            n_dissections=20,
            statistical_analysis=True,
            splt_analysis=0,
            n_meters=0,
            max_nflows=0,
            performance_report=0,
            system_visibility_mode=0,
            system_visibility_poll_ms=100
        )

        extracted_flows = streamer.to_pandas()
        
        if extracted_flows is None:
            print(f"Warning: No flows extracted from {pcap_path.name}. Skipping...")
            continue      
  
        extracted_flows['activity'] = 'Unknown'
        extracted_flows['stage'] = 'Unknown'
        
        for extracted_flow in extracted_flows.itertuples():
            key = (extracted_flow.src_ip, extracted_flow.src_port, extracted_flow.dst_ip, extracted_flow.dst_port, extracted_flow.protocol)
            
            flow_text = '';
            flow_text += f'[{extracted_flow.protocol:02}]'
            flow_text += f' {extracted_flow.src_ip}:{extracted_flow.src_port} -> {extracted_flow.dst_ip}:{extracted_flow.dst_port}'
                        
            extracted_start_time = datetime.datetime.fromtimestamp(extracted_flow.bidirectional_first_seen_ms / 1000) # type: ignore
            extracted_stop_time = datetime.datetime.fromtimestamp(extracted_flow.bidirectional_last_seen_ms / 1000) # type: ignore

            candidate_flows = dapt2020_map.get(key, [])
            candidate_labels = set()
            
            for candidate_timestamp, candidate_duration, candidate_activity, candidate_stage in candidate_flows:
                candidate_start_time = candidate_timestamp
                candidate_stop_time = candidate_timestamp + pd.Timedelta(milliseconds=candidate_duration)                
                
                if (extracted_start_time <= candidate_stop_time) and (extracted_stop_time >= candidate_start_time):
                    candidate_labels.add((candidate_activity, candidate_stage))
            
            if len(candidate_labels) == 2 and ('Normal', 'Benign') in candidate_labels:
                candidate_labels.remove(('Normal', 'Benign'))
            
            if len(candidate_labels) == 1:
                assigned_activity, assigned_stage = candidate_labels.pop()
                extracted_flows.at[extracted_flow.Index, 'activity'] = assigned_activity
                extracted_flows.at[extracted_flow.Index, 'stage'] = assigned_stage
            elif len(candidate_labels) > 1:
                print(f'  {flow_text}')
                print(f'    => Multiple conflicting labels found: {candidate_labels}.')
            else:
                print(f'  {flow_text}')
                print('    => No matching labels found.')
        
        output_flow_path = output_path.joinpath(pcap_path.name + '.csv')
        extracted_flows.to_csv(output_flow_path, index=False)
        