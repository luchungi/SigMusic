import os
import shutil
import numpy as np
import pandas as pd
import pretty_midi

def midi_to_df(midi_data):
    midi_list = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            start = note.start
            end = note.end
            pitch = note.pitch
            velocity = note.velocity
            midi_list.append([start, end, pitch, velocity, instrument.name])

    midi_list = sorted(midi_list, key=lambda x: (x[0], x[2]))

    df = pd.DataFrame(midi_list, columns=['Start', 'End', 'Pitch', 'Velocity', 'Instrument'])
    return df

def df_to_midi(df, instrument_name=None):
    midi_data = pretty_midi.PrettyMIDI()
    if instrument_name is None:
        instrument = pretty_midi.Instrument(program=0)
    else:
        program = pretty_midi.instrument_name_to_program(instrument_name)
        instrument = pretty_midi.Instrument(program=program)

    for _, row in df.iterrows():
        start = row['Start']
        end = row['End']
        pitch = row['Pitch']
        velocity = row['Velocity']
        note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end)
        instrument.notes.append(note)
    midi_data.instruments.append(instrument)
    return midi_data

def midi_to_list(midi):
    """Convert a midi file to a list of note events

    Notebook: C1/C1S2_MIDI.ipynb

    Args:
        midi (str or pretty_midi.pretty_midi.PrettyMIDI): Either a path to a midi file or PrettyMIDI object

    Returns:
        score (list): A list of note events where each note is specified as
            ``[start, duration, pitch, velocity, label]``
    """

    if isinstance(midi, str):
        midi_data = pretty_midi.pretty_midi.PrettyMIDI(midi)
    elif isinstance(midi, pretty_midi.pretty_midi.PrettyMIDI):
        midi_data = midi
    else:
        raise RuntimeError('midi must be a path to a midi file or pretty_midi.PrettyMIDI')

    score = []

    for instrument in midi_data.instruments:
        for note in instrument.notes:
            start = note.start
            duration = note.end - start
            pitch = note.pitch
            velocity = note.velocity / 128.
            score.append([start, duration, pitch, velocity, instrument.name])
    return score

def add_octave_and_note(df, midi_note_df, inplace=False):
    '''
    Find the octave and note of each pitch in df by matching it to the data in midi_note_df and add it to the dataframe
    '''
    df = df.copy() if not inplace else df
    for i, row in df.iterrows():
        octave = midi_note_df.index[midi_note_df[midi_note_df == int(row['Pitch'])].notna().sum(axis=1).astype(bool)]
        if len(octave) > 1:
            raise ValueError('More than one octave found')
        else:
            df.at[i, 'octave'] = octave[0]
        note = midi_note_df.columns[midi_note_df[midi_note_df == int(row['Pitch'])].notna().sum(axis=0).astype(bool)]
        if len(note) > 1:
            raise ValueError('More than one note found')
        else:
            df.at[i, 'note'] = note[0]
    if not inplace:
        return df

def get_key_note(key_number, tight=False):
    if key_number < 0 or key_number > 23:
        raise ValueError('Key number must be between 0 and 23.')
    if tight:
        notes = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
        scales = ['', 'm']
    else:
        notes = ['C', 'C#/Db', 'D', 'D#/Eb', 'E', 'F', 'F#/Gb', 'G', 'G#/Ab', 'A', 'A#/Bb', 'B']
        scales = ['major', 'minor']
    note = notes[key_number % 12]
    scale = scales[key_number // 12]
    if tight:
        return f'{note}{scale}'
    else:
        return f'{note} {scale}'

def copy_to_single_key(entry, key_number):
    path = entry.path.split('/')
    if len(path) != 8:
        raise ValueError('Path does not conform to expected format (length).')
    new_path = f'./data/single_key/{key_number}/{path[4]}/{path[5]}/{path[6]}/{path[7]}'
    if not os.path.exists(f'./data/single_key/{key_number}/{path[4]}/{path[5]}/{path[6]}/'):
        os.makedirs(f'./data/single_key/{key_number}/{path[4]}/{path[5]}/{path[6]}/')
    # print(new_path)
    shutil.copy(entry.path, new_path)

def copy_to_multiple_keys(entry):
    path = entry.path.split('/')
    if len(path) != 8:
        raise ValueError('Path does not conform to expected format (length).')
    new_path = f'./data/multiple_keys/{path[4]}/{path[5]}/{path[6]}/{path[7]}'
    if not os.path.exists(f'./data/multiple_keys/{path[4]}/{path[5]}/{path[6]}/'):
        os.makedirs(f'./data/multiple_keys/{path[4]}/{path[5]}/{path[6]}/')
    # print(new_path)
    shutil.copy(entry.path, new_path)

def no_key_change_in_song(keys):
    return keys.count(keys[0]) == len(keys)

def get_key_and_sort_files_to_dir(directory: str, key_counts: np.ndarray, n_midi_files: int=0):
    if n_midi_files > 0: # multiple midi files for same song
        files = []
        keys = []
        for entry in os.scandir(directory):
            if entry.is_file() and entry.name.endswith('.mid'):
                files.append(entry)
                midi_data = pretty_midi.PrettyMIDI(entry.path)
                key_signature = midi_data.key_signature_changes
                if len(key_signature) > 1:
                    raise ValueError(f'More than one key signature change in midi file {entry.path}')
                keys.append(int(key_signature[0].key_number))
        if no_key_change_in_song(keys):
            # print(f'{directory.split("/")[6]} written in {get_key_note(keys[0])} key')
            key_counts[keys[0]] += 1
            for file in files:
                copy_to_single_key(file, keys[0])
        else:
            # unique_keys = set(keys)
            # key_notes = [get_key_note(key) for key in unique_keys]
            # print(f'{directory.split("/")[6]} written in {len(unique_keys)} keys: {key_notes}')
            for file in files:
                copy_to_multiple_keys(file)
    else:
        for entry in os.scandir(directory):
            if entry.is_dir():
                n_midi_files = len([name for name in os.listdir(entry.path) if name.endswith('.mid')])
                get_key_and_sort_files_to_dir(entry.path, key_counts, n_midi_files)