from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

                                                                                                                      

def get_trellis(emission, tokens, blank_id=-1):
    num_frame = emission.shape[0]
    num_tokens = len(tokens)

                                                                
                                                                   
                                                                    
    trellis = np.empty((num_frame + 1, num_tokens + 1))
    trellis[0, 0] = 0
    trellis[1:, 0] = np.cumsum(emission[:, blank_id], 0)
    trellis[0, -num_tokens:] = -float("inf")
    trellis[-num_tokens:, 0] = float("inf")

    for t in range(num_frame):
        trellis[t + 1, 1:] = np.maximum(
                                                 
            trellis[t, 1:] + emission[t, blank_id],
                                                  
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


def backtrack(trellis, emission, tokens, blank_id=-1):
           
                                                                 
                                           
                                                        
                                                   
                                                              
                                                     
    j = trellis.shape[1] - 1
    t_start = np.argmax(trellis[:, j])

    path = []
    for t in range(t_start, 0, -1):
                                                                  
                       
                                                                                 
                                                                    
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
                                                             
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

                                                        
        prob = np.exp(emission[t - 1, tokens[j - 1] if changed > stayed else blank_id]).item()
                                                                      
        path.append(Point(j - 1, t - 1, prob))

                             
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        raise ValueError("Failed to align")
    return path[::-1]

                  
@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


def merge_repeats(path, transcript, exclude_blank=False):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        if not exclude_blank:
            score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
            segments.append(
                Segment(
                    transcript[path[i1].token_index],
                    path[i1].time_index,
                    path[i2 - 1].time_index + 1,
                    score,
                )
            )
        else:
            score = path[i1].score
            segments.append(
                Segment(
                    transcript[path[i1].token_index],
                    path[i1].time_index,
                    path[i1].time_index + 1,
                    score
                )
            )
        i1 = i2
    return segments

def plot_trellis_with_segments(trellis, segments, transcript):
                                                                 
    trellis_with_path = np.copy(trellis)
    for i, seg in enumerate(segments):
        if seg.label != "|":
            trellis_with_path[seg.start + 1 : seg.end + 1, i + 1] = float("nan")

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 9.5))
    ax1.set_title("Path, label and probability for each label")
    ax1.imshow(trellis_with_path.T, origin="lower")
    ax1.set_xticks([])

    for i, seg in enumerate(segments):
        if seg.label != "|":
            ax1.annotate(seg.label, (seg.start + 0.7, i + 0.3), weight="bold")
            ax1.annotate(f"{seg.score:.2f}", (seg.start - 0.3, i + 4.3))

def plot_trellis_with_path(trellis, path):
                                                                 
    trellis_with_path = np.copy(trellis)
    for _, p in enumerate(path):
        trellis_with_path[p.time_index, p.token_index] = float("nan")
    plt.imshow(trellis_with_path[1:, 1:].T, origin="lower")
