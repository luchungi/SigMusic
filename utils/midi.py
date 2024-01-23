import pretty_midi
import pandas as pd
import libfmp.c1

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