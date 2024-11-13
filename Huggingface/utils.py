# import chord_recognition
import numpy as np
import miditoolkit
import os
import scipy.stats
from scipy.io import loadmat
import numpy as np
import pandas as pd

# parameters for input
DEFAULT_VELOCITY_BINS = np.linspace(0, 128, 32+1, dtype=np.int)
DEFAULT_FRACTION = 16
DEFAULT_DURATION_BINS = np.arange(60, 3841, 60, dtype=int)
DEFAULT_TEMPO_INTERVALS = [range(30, 90), range(90, 150), range(150, 210)]

# parameters for output
DEFAULT_RESOLUTION = 480

# define "Item" for general storage
class Item(object):
    def __init__(self, name, start, end, velocity, pitch):
        self.name = name
        self.start = start
        self.end = end
        self.velocity = velocity
        self.pitch = pitch

    def __repr__(self):
        return 'Item(name={}, start={}, end={}, velocity={}, pitch={})'.format(
            self.name, self.start, self.end, self.velocity, self.pitch)

# read notes and tempo changes from midi (assume there is only one track)
def read_items(file_path):
    midi_obj = miditoolkit.midi.parser.MidiFile(file_path)
    # note
    note_items = []
    notes = midi_obj.instruments[0].notes
    notes.sort(key=lambda x: (x.start, x.pitch))
    for note in notes:
        note_items.append(Item(
            name='Note', 
            start=note.start, 
            end=note.end, 
            velocity=note.velocity, 
            pitch=note.pitch))
    note_items.sort(key=lambda x: x.start)
    # tempo
    tempo_items = []
    for tempo in midi_obj.tempo_changes:
        tempo_items.append(Item(
            name='Tempo',
            start=tempo.time,
            end=None,
            velocity=None,
            pitch=int(tempo.tempo)))
    tempo_items.sort(key=lambda x: x.start)
    # expand to all beat
    max_tick = tempo_items[-1].start
    existing_ticks = {item.start: item.pitch for item in tempo_items}
    wanted_ticks = np.arange(0, max_tick+1, DEFAULT_RESOLUTION)
    output = []
    for tick in wanted_ticks:
        if tick in existing_ticks:
            output.append(Item(
                name='Tempo',
                start=tick,
                end=None,
                velocity=None,
                pitch=existing_ticks[tick]))
        else:
            output.append(Item(
                name='Tempo',
                start=tick,
                end=None,
                velocity=None,
                pitch=output[-1].pitch))
    tempo_items = output
    return note_items, tempo_items

# quantize items
def quantize_items(items, ticks=120):
    # grid
    grids = np.arange(0, items[-1].start, ticks, dtype=int)
    # process
    for item in items:
        index = np.argmin(abs(grids - item.start))
        shift = grids[index] - item.start
        item.start += shift
        item.end += shift
    return items      

# extract chord
def extract_chords(items):
    '''
    method = chord_recognition.MIDIChord()
    chords = method.extract(notes=items)
    output = []
    for chord in chords:
        output.append(Item(
            name='Chord',
            start=chord[0],
            end=chord[1],
            velocity=None,
            pitch=chord[2].split('/')[0]))
    '''
    output = []
    return output

# group items
def group_items(items, max_time, ticks_per_bar=DEFAULT_RESOLUTION*4):
    items.sort(key=lambda x: x.start)
    downbeats = np.arange(0, max_time+ticks_per_bar, ticks_per_bar)
    groups = []
    for db1, db2 in zip(downbeats[:-1], downbeats[1:]):
        insiders = []
        for item in items:
            if (item.start >= db1) and (item.start < db2):
                insiders.append(item)
        overall = [db1] + insiders + [db2]
        groups.append(overall)
    return groups

# define "Event" for event storage
class Event(object):
    def __init__(self, name, time, value, text):
        self.name = name
        self.time = time
        self.value = value
        self.text = text

    def __repr__(self):
        return 'Event(name={}, time={}, value={}, text={})'.format(
            self.name, self.time, self.value, self.text)

# item to event
def item2event(groups):
    events = []
    n_downbeat = 0
    for i in range(len(groups)):
        if 'Note' not in [item.name for item in groups[i][1:-1]]:
            continue
        bar_st, bar_et = groups[i][0], groups[i][-1]
        n_downbeat += 1
        events.append(Event(
            name='Bar',
            time=None, 
            value=None,
            text='{}'.format(n_downbeat)))
        for item in groups[i][1:-1]:
            # position
            flags = np.linspace(bar_st, bar_et, DEFAULT_FRACTION, endpoint=False)
            index = np.argmin(abs(flags-item.start))
            events.append(Event(
                name='Position', 
                time=item.start,
                value='{}/{}'.format(index+1, DEFAULT_FRACTION),
                text='{}'.format(item.start)))
            if item.name == 'Note':
                # velocity
                velocity_index = np.searchsorted(
                    DEFAULT_VELOCITY_BINS, 
                    item.velocity, 
                    side='right') - 1
                events.append(Event(
                    name='Note Velocity',
                    time=item.start, 
                    value=velocity_index,
                    text='{}/{}'.format(item.velocity, DEFAULT_VELOCITY_BINS[velocity_index])))
                # pitch
                events.append(Event(
                    name='Note On',
                    time=item.start, 
                    value=item.pitch,
                    text='{}'.format(item.pitch)))
                # duration
                duration = item.end - item.start
                index = np.argmin(abs(DEFAULT_DURATION_BINS-duration))
                events.append(Event(
                    name='Note Duration',
                    time=item.start,
                    value=index,
                    text='{}/{}'.format(duration, DEFAULT_DURATION_BINS[index])))
            elif item.name == 'Chord':
                events.append(Event(
                    name='Chord', 
                    time=item.start,
                    value=item.pitch,
                    text='{}'.format(item.pitch)))
            elif item.name == 'Tempo':
                tempo = item.pitch
                if tempo in DEFAULT_TEMPO_INTERVALS[0]:
                    tempo_style = Event('Tempo Class', item.start, 'slow', None)
                    tempo_value = Event('Tempo Value', item.start, 
                        tempo-DEFAULT_TEMPO_INTERVALS[0].start, None)
                elif tempo in DEFAULT_TEMPO_INTERVALS[1]:
                    tempo_style = Event('Tempo Class', item.start, 'mid', None)
                    tempo_value = Event('Tempo Value', item.start, 
                        tempo-DEFAULT_TEMPO_INTERVALS[1].start, None)
                elif tempo in DEFAULT_TEMPO_INTERVALS[2]:
                    tempo_style = Event('Tempo Class', item.start, 'fast', None)
                    tempo_value = Event('Tempo Value', item.start, 
                        tempo-DEFAULT_TEMPO_INTERVALS[2].start, None)
                elif tempo < DEFAULT_TEMPO_INTERVALS[0].start:
                    tempo_style = Event('Tempo Class', item.start, 'slow', None)
                    tempo_value = Event('Tempo Value', item.start, 0, None)
                elif tempo > DEFAULT_TEMPO_INTERVALS[2].stop:
                    tempo_style = Event('Tempo Class', item.start, 'fast', None)
                    tempo_value = Event('Tempo Value', item.start, 59, None)
                events.append(tempo_style)
                events.append(tempo_value)     
    return events

#############################################################################################
# WRITE MIDI
#############################################################################################
def word_to_event(words, word2event):
    events = []
    for word in words:
        event_name, event_value = word2event.get(word).split('_')
        events.append(Event(event_name, None, event_value, None))
    return events

def write_midi(words, word2event, output_path, prompt_path=None):
    events = word_to_event(words, word2event)
    # get downbeat and note (no time)
    temp_notes = []
    temp_chords = []
    temp_tempos = []
    for i in range(len(events)-3):
        if events[i].name == 'Bar' and i > 0:
            temp_notes.append('Bar')
            temp_chords.append('Bar')
            temp_tempos.append('Bar')
        elif events[i].name == 'Position' and \
            events[i+1].name == 'Note Velocity' and \
            events[i+2].name == 'Note On' and \
            events[i+3].name == 'Note Duration':
            # start time and end time from position
            position = int(events[i].value.split('/')[0]) - 1
            # velocity
            index = int(events[i+1].value)
            velocity = int(DEFAULT_VELOCITY_BINS[index])
            # pitch
            pitch = int(events[i+2].value)
            # duration
            index = int(events[i+3].value)
            duration = DEFAULT_DURATION_BINS[index]
            # adding
            temp_notes.append([position, velocity, pitch, duration])
        elif events[i].name == 'Position' and events[i+1].name == 'Chord':
            position = int(events[i].value.split('/')[0]) - 1
            temp_chords.append([position, events[i+1].value])
        elif events[i].name == 'Position' and \
            events[i+1].name == 'Tempo Class' and \
            events[i+2].name == 'Tempo Value':
            position = int(events[i].value.split('/')[0]) - 1
            if events[i+1].value == 'slow':
                tempo = DEFAULT_TEMPO_INTERVALS[0].start + int(events[i+2].value)
            elif events[i+1].value == 'mid':
                tempo = DEFAULT_TEMPO_INTERVALS[1].start + int(events[i+2].value)
            elif events[i+1].value == 'fast':
                tempo = DEFAULT_TEMPO_INTERVALS[2].start + int(events[i+2].value)
            temp_tempos.append([position, tempo])
    # get specific time for notes
    ticks_per_beat = DEFAULT_RESOLUTION
    ticks_per_bar = DEFAULT_RESOLUTION * 4 # assume 4/4
    notes = []
    current_bar = 0
    for note in temp_notes:
        if note == 'Bar':
            current_bar += 1
        else:
            position, velocity, pitch, duration = note
            # position (start time)
            current_bar_st = current_bar * ticks_per_bar
            current_bar_et = (current_bar + 1) * ticks_per_bar
            flags = np.linspace(current_bar_st, current_bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int)
            st = flags[position]
            # duration (end time)
            et = st + duration
            notes.append(miditoolkit.Note(velocity, pitch, st, et))
    # get specific time for chords
    if len(temp_chords) > 0:
        chords = []
        current_bar = 0
        for chord in temp_chords:
            if chord == 'Bar':
                current_bar += 1
            else:
                position, value = chord
                # position (start time)
                current_bar_st = current_bar * ticks_per_bar
                current_bar_et = (current_bar + 1) * ticks_per_bar
                flags = np.linspace(current_bar_st, current_bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int)
                st = flags[position]
                chords.append([st, value])
    # get specific time for tempos
    tempos = []
    current_bar = 0
    for tempo in temp_tempos:
        if tempo == 'Bar':
            current_bar += 1
        else:
            position, value = tempo
            # position (start time)
            current_bar_st = current_bar * ticks_per_bar
            current_bar_et = (current_bar + 1) * ticks_per_bar
            flags = np.linspace(current_bar_st, current_bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int)
            st = flags[position]
            tempos.append([int(st), value])
    # write
    if prompt_path:
        midi = miditoolkit.midi.parser.MidiFile(prompt_path)
        #
        last_time = DEFAULT_RESOLUTION * 4 * 4
        # note shift
        for note in notes:
            note.start += last_time
            note.end += last_time
        midi.instruments[0].notes.extend(notes)
        # tempo changes
        temp_tempos = []
        for tempo in midi.tempo_changes:
            if tempo.time < DEFAULT_RESOLUTION*4*4:
                temp_tempos.append(tempo)
            else:
                break
        for st, bpm in tempos:
            st += last_time
            temp_tempos.append(miditoolkit.midi.containers.TempoChange(bpm, st))
        midi.tempo_changes = temp_tempos
        # write chord into marker
        if len(temp_chords) > 0:
            for c in chords:
                midi.markers.append(
                    miditoolkit.midi.containers.Marker(text=c[1], time=c[0]+last_time))
    else:
        midi = miditoolkit.midi.parser.MidiFile()
        midi.ticks_per_beat = DEFAULT_RESOLUTION
        # write instrument
        inst = miditoolkit.midi.containers.Instrument(0, is_drum=False)
        inst.notes = notes
        midi.instruments.append(inst)
        # write tempo
        tempo_changes = []
        for st, bpm in tempos:
            tempo_changes.append(miditoolkit.midi.containers.TempoChange(bpm, st))
        midi.tempo_changes = tempo_changes
        # write chord into marker
        if len(temp_chords) > 0:
            for c in chords:
                midi.markers.append(
                    miditoolkit.midi.containers.Marker(text=c[1], time=c[0]))
    # write
    midi.dump(output_path)

N_PITCH_CLS = 12 # {C, C#, ..., Bb, B}

def get_event_seq(piece_csv, seq_col_name='ENCODING'):
  '''
  Extracts the event sequence from a piece of music (stored in .csv file).
  NOTE: You should modify this function if you use different formats.

  Parameters:
    piece_csv (str): path to the piece's .csv file.
    seq_col_name (str): name of the column containing event encodings.

  Returns:
    list: the event sequence of the piece.
  '''
  df = pd.read_csv(piece_csv, encoding='utf-8')
  return df[seq_col_name].astype('int32').tolist()


def get_chord_sequence(ev_seq, chord_evs):
  '''
  Extracts the chord sequence (in string representation) from the input piece.
  NOTE: This function is vocabulary-dependent, 
        you should implement a new one if a different vocab is used. 

  Parameters:
    ev_seq (list): a piece of music in event sequence representation.
    chord_evs (dict of lists): [key] type of chord-related event --> [value] encodings belonging to the type.

  Returns:
    list of lists: The chord sequence of the input piece, each element (a list) being the representation of a single chord.
  '''
  # extract chord-related tokens
  ev_seq = [
    x for x in ev_seq if any(x in chord_evs[typ] for typ in chord_evs.keys())
  ]

  # remove grammar errors in sequence (vocabulary-dependent)
  legal_seq = []
  cnt = 0
  for i, ev in enumerate(ev_seq):
    cnt += 1
    if ev in chord_evs['Chord-Slash'] and cnt == 3:
      cnt = 0
      legal_seq.extend(ev_seq[i-2:i+1])
  
  ev_seq = legal_seq
  assert not len(ev_seq) % 3
  chords = []
  for i in range(0, len(ev_seq), 3):
    chords.append( ev_seq[i:i+3] )

  return chords

def compute_histogram_entropy(hist):
  ''' 
  Computes the entropy (log base 2) of a normalised histogram.

  Parameters:
    hist (ndarray): input pitch (or duration) histogram, should be normalised.

  Returns:
    float: entropy (log base 2) of the histogram.
  '''
  return scipy.stats.entropy(hist) / np.log(2)


def get_pitch_histogram(ev_seq, pitch_evs=range(128), verbose=False):
  '''
  Computes the pitch-class histogram from an event sequence.

  Parameters:
    ev_seq (list): a piece of music in event sequence representation.
    pitch_evs (list): encoding IDs of ``Note-On`` events, should be sorted in increasing order by pitches.
    verbose (bool): whether to print msg. when ev_seq has no notes.

  Returns:
    ndarray: the resulting pitch-class histogram.
  '''
  ev_seq = [x for x in ev_seq if x in pitch_evs]

  if not len(ev_seq):
    if verbose:
      print ('[Info] The sequence contains no notes.')
    return None

  # compress sequence to pitch classes & get normalised counts
  ev_seq = pd.Series(ev_seq) % N_PITCH_CLS
  ev_hist = ev_seq.value_counts(normalize=True)

  # make the final histogram
  hist = np.zeros( (N_PITCH_CLS,) )
  for i in range(N_PITCH_CLS):
    if i in ev_hist.index:
      hist[i] = ev_hist.loc[i]

  return hist

def get_onset_xor_distance(seq_a, seq_b, bar_ev_id, pos_evs, pitch_evs=range(128)):
  '''
  Computes the XOR distance of onset positions between a pair of bars.
  
  Parameters:
    seq_a, seq_b (list): event sequence of a bar of music.
      IMPORTANT: for this implementation, a ``Note-Position`` event must appear before the associated ``Note-On``.
    bar_ev_id (int): encoding ID of the ``Bar`` event, vocabulary-dependent.
    pos_evs (list): encoding IDs of ``Note-Position`` events, vocabulary-dependent.
    pitch_evs (list): encoding IDs of ``Note-On`` events.

  Returns:
    float: 0~1, the XOR distance between the 2 bars' (seq_a, seq_b) binary vectors of onsets.
  '''
  # sanity checks
  assert seq_a[0] == bar_ev_id and seq_b[0] == bar_ev_id
  assert seq_a.count(bar_ev_id) == 1 and seq_b.count(bar_ev_id) == 1

  # compute binary onset vectors
  n_pos = len(pos_evs)
  def make_onset_vec(seq):
    cur_pos = -1
    onset_vec = np.zeros((n_pos,))
    for ev in seq:
      if ev in pos_evs:
        cur_pos = ev - pos_evs[0]
      if ev in pitch_evs:
        onset_vec[cur_pos] = 1
    return onset_vec
  a_onsets, b_onsets = make_onset_vec(seq_a), make_onset_vec(seq_b)

  # compute XOR distance
  dist = np.sum( np.abs(a_onsets - b_onsets) ) / n_pos
  return dist

def get_bars_crop(ev_seq, start_bar, end_bar, bar_ev_id, verbose=False):
  '''
  Returns the designated crop (bars) of the input piece.

  Parameter:
    ev_seq (list): a piece of music in event sequence representation.
    start_bar (int): the starting bar of the crop.
    end_bar (int): the ending bar (inclusive) of the crop.
    bar_ev_id (int): encoding ID of the ``Bar`` event, vocabulary-dependent.
    verbose (bool): whether to print messages when unexpected operations happen.

  Returns:
    list: a cropped segment of music consisting of (end_bar - start_bar + 1) bars.
  '''
  if start_bar < 0 or end_bar < 0:
    raise ValueError('Invalid start_bar: {}, or end_bar: {}.'.format(start_bar, end_bar))

  # get the indices of ``Bar`` events
  ev_seq = np.array(ev_seq)
  bar_markers = np.where(ev_seq == bar_ev_id)[0]

  if start_bar > len(bar_markers) - 1:
    raise ValueError('start_bar: {} beyond end of piece.'.format(start_bar))

  if end_bar < len(bar_markers) - 1:
    cropped_seq = ev_seq[ bar_markers[start_bar] : bar_markers[end_bar + 1] ]
  else:
    if verbose:
      print (
        '[Info] end_bar: {} beyond or equal the end of the input piece; only the last {} bars are returned.'.format(
          end_bar, len(bar_markers) - start_bar
        ))
    cropped_seq = ev_seq[ bar_markers[start_bar] : ]

  return cropped_seq.tolist()

def read_fitness_mat(fitness_mat_file):
  '''
  Reads and returns (as an ndarray) a fitness scape plot as a center-duration matrix.

  Parameters:
    fitness_mat_file (str): path to the file containing fitness scape plot.
      Accepted formats: .mat (MATLAB data), .npy (ndarray)

  Returns:
    ndarray: the fitness scapeplot encoded as a center-duration matrix.
  '''
  ext = os.path.splitext(fitness_mat_file)[-1].lower()

  if ext == '.npy':
    f_mat = np.load(fitness_mat_file)
  elif ext == '.mat':
    mat_dict = loadmat(fitness_mat_file)
    f_mat = mat_dict['fitness_info'][0, 0][0]
    f_mat[ np.isnan(f_mat) ] = 0.0
  else:
    raise ValueError('Unsupported fitness scape plot format: {}'.format(ext))

  for slen in range(f_mat.shape[0]):
    f_mat[slen] = np.roll(f_mat[slen], slen // 2)

  return f_mat
