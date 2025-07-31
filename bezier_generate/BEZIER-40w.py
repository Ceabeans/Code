# Palmprint synthesize in the 2D palne.
# The original implementation was written by Kai Zhao (kz@kaizhao.net) in Tencent, this is a reimplementation by Kai Zhao
# with confidential content removed.
# This reimplementation is only for research purpose, commercial use of this code must be officially permitted by Tencent.
# Copyright: Tencent
import bezier, mmcv
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from multiprocessing import Pool

from vlkit.image import norm255
from vlkit.utils import AverageMeter

import os, sys, argparse, glob, cv2, random, time
from os.path import join, split, isdir, isfile, dirname
from copy import copy
from tqdm import tqdm
from PIL import Image, ImageFilter, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args():
    parser = argparse.ArgumentParser(description='Compare results')     
    parser.add_argument('--num_ids', type=int, default=4000)             
    parser.add_argument('--samples', type=int, default=100)              
    parser.add_argument('--nproc', type=int, default=100)                 
    parser.add_argument('--imsize', type=int, default=256)              
    parser.add_argument('--output', type=str, default='./BEZIER-40w')    
    args = parser.parse_args()
    assert args.num_ids % args.nproc == 0   
    return args


def sample_edge(low, high):
    """
    sample points on edges of a unit square 
    """
    offset = min(low, high)
    low, high = map(lambda x: x - offset, [low, high])  
    t = np.random.uniform(low, high) + offset   

    if t >= 4:
        t = t % 4
    if t < 0:
        t = t + 4

    if t <= 1:
        x, y = t, 0
    elif 1 < t <= 2:
        x, y = 1, t - 1
    elif 2 < t <= 3:
        x, y = 3 - t, 1
    else:
        x, y = 0, 4 - t
    return np.array([x, y]), t


def control_point(head, tail, t=0.5, s=0):      
    head = np.array(head)
    tail = np.array(tail)
    l = np.sqrt(((head - tail) ** 2).sum())     
    assert head.size == 2 and tail.size == 2    
    assert l >= 0
    c = head * t + (1 - t) * tail               
    x, y = head - tail
    v = np.array([-y, x])
    v /= max(np.sqrt((v ** 2).sum()), 1e-6)     
    return c + s * l * v                        


def get_bezier(p0, p1, t=0.5, s=1):
    assert -1 < s < 1, 's=%f'%s             
    c = control_point(p0, p1, t, s)
    nodes = np.vstack((p0, c, p1)).T
    return bezier.Curve(nodes, degree=2)


def generate_parameters_1():                          
    # 起点\终点 &t
    head1, thead1 = sample_edge(0, 0.3)             # (0,0)-（0.3，0）
    tail1, ttail1 = sample_edge(1.2, 1.6)           #(1,0.2)-（1,0.6）

    if np.random.uniform() <= 0.5:
        head2, thead2 = sample_edge(-0.5, -0.25)    
        head3, thead3 = head2, thead2               
    else:
        head2, thead2 = sample_edge(-0.23, -0.08)   #(0,0.08)-（0,0.23）
        head3, thead3 = sample_edge(-0.5, -0.25)    #(0,0.25)-（0,0.5）
    tail2, ttail2 = sample_edge(1.8, 2.15)          #(0.85,1)-（1,0.8）
    tail3, ttail3 = sample_edge(2.3, 2.7)           #(0.3,1)-（0.7,1）

    if np.random.uniform() <= 0.7:
        head4, thead4 = sample_edge(0, 0.2)         #(0,0)-(0.2,0)
        if ttail3 <= 2.5:                           
            tail4, ttail4 = sample_edge(2.1, 2.5)   # (0.5,1)-(0.9,1)
        else:
            tail4, ttail4 = sample_edge(2.5, 2.7)   # (0.3,1)-(0.5,1)
    else:
        head4, thead4 = sample_edge(2.4, 3)         
        tail4, ttail4 = sample_edge(1.2, 1.8)       


    c1 = control_point(head1, tail1, s=-np.random.uniform(0.13, 0.16))
    c2 = control_point(head2, tail2, s=np.random.uniform(0.1, 0.2))
    if ttail3 <= 2.5:                               
        c3 = control_point(head3, tail3, s=np.random.uniform(0.15, 0.2))
    else:
        c3 = control_point(head3, tail3, s=np.random.uniform(0.1, 0.15))

    c4 = control_point(head4, tail4, s=np.random.uniform(0.05, 0.15))

    return np.vstack((head1, c1, tail1)), np.vstack((head2, c2, tail2)), np.vstack((head3, c3, tail3)), np.vstack((head4, c4, tail4))


def generate_parameters_2():                        

    head1, thead1 = sample_edge(0, 0.3)             # (0,0)-（0.3，0）
    tail1, ttail1 = sample_edge(1.2, 1.6)           #(1,0.2)-（1,0.6）

    if np.random.uniform() <= 0.5:
        head2, thead2 = sample_edge(-0.5, -0.25)    
        head3, thead3 = head2, thead2               
    else:
        head2, thead2 = sample_edge(-0.23, -0.08)   #(0,0.08)-（0,0.23）
        head3, thead3 = sample_edge(-0.5, -0.25)    #(0,0.25)-（0,0.5）
    tail2, ttail2 = sample_edge(1.8, 2.15)          #(0.85,1)-（1,0.8）
    tail3, ttail3 = sample_edge(2.3, 2.7)           #(0.3,1)-（0.7,1）

    if np.random.uniform() <= 0.7:
        head4, thead4 = sample_edge(0, 0.2)         #(0,0)-(0.2,0)
        if ttail3 <= 2.5:                           
            tail4, ttail4 = sample_edge(2.1, 2.5)   # (0.5,1)-(0.9,1)
        else:
            tail4, ttail4 = sample_edge(2.5, 2.7)   # (0.3,1)-(0.5,1)
    else:
        head4, thead4 = sample_edge(2.4, 3)         #(0,1)-(0.6,1)
        tail4, ttail4 = sample_edge(1.2, 1.8)       #(1,0.2)-(1,0.8)


    c11 = control_point(head1, tail1, t=np.random.uniform(0.3, 0.7), s=-np.random.uniform(0.1, 0.2))
    c12 = c11
    c21 = control_point(head2, tail2, t=0.3, s=np.random.uniform(0.01, 0.1))
    c22 = control_point(head2, tail2, t=0.7, s=-np.random.uniform(0.01, 0.05))
    c31 = control_point(head3, tail3, t=np.random.uniform(0.4, 0.6), s=np.random.uniform(0.1, 0.2))
    c32 = c31
    c41 = control_point(head4, tail4, t=0.3, s=-np.random.uniform(0.05, 0.15))
    c42 = control_point(head4, tail4, t=0.7, s=np.random.uniform(0.05, 0.15))

    return np.vstack((head1, c11, c12, tail1)), np.vstack((head2, c21, c22, tail2)), np.vstack((head3, c31, c32, tail3)), np.vstack((head4, c41, c42, tail4))


def generate_parameters_3():                        
    # 起点\终点 &t
    if np.random.uniform() <= 0.5:
        head1, thead1 = sample_edge(-0.5, -0.2)     
        head2, thead2 = head1, thead1
    else:
        head1, thead1 = sample_edge(-0.25, -0.2)    #(0,0.2)-（0,0.25）
        head2, thead2 = sample_edge(-0.6, -0.3)     #(0,0.3)-（0,0.6）

    tail1, ttail1 = sample_edge(1.2, 1.6)           #（1,0.2）-（1,0.6）
    tail2, ttail2 = sample_edge(2.3, 2.7)           #(0.3,1)-（0.7,1）

    if np.random.uniform() <= 0.7:
        head3, thead3 = sample_edge(0, 0.2)         #(0,0)-(0.2,0)
        if ttail2 <= 2.5:                           
            tail3, ttail3 = sample_edge(2.1, 2.5)   # (0.5,1)-(0.9,1)
        else:
            tail3, ttail3 = sample_edge(2.5, 2.7)   # (0.3,1)-(0.5,1)
    else:
        head3, thead3 = sample_edge(2.4, 3)         #(0,1)-(0.6,1)
        tail3, ttail3 = sample_edge(1.2, 1.8)       #(1,0.2)-(1,0.8)

    c1 = control_point(head1, tail1, s=np.random.uniform(0.01, 0.05))
    if ttail2 <= 2.5:                               
        c2 = control_point(head2, tail2, s=np.random.uniform(0.15, 0.2))
    else:
        c2 = control_point(head2, tail2, s=np.random.uniform(0.1, 0.15))

    c3 = control_point(head3, tail3, s=np.random.uniform(0.05, 0.15))

    return np.vstack((head1, c1, tail1)), np.vstack((head2, c2, tail2)), np.vstack((head3, c3, tail3))


def batch_process(proc_index, ranges, args):
    ids_per_proc = int(args.num_ids / args.nproc)   
    EPS = 1e-2
    np.random.seed(proc_index)
    random.seed(proc_index)
    average_meter = AverageMeter(name='time')
    local_idx = 0

    for id_idx, i in enumerate(range(*ranges[proc_index])):
        tic = time.time()

        nodes2 = None  

        if np.random.uniform() <= 0.5:                                          
            nodes1 = generate_parameters_1()
            flag1 = [np.random.uniform() > 0.00001, np.random.uniform() > 0.00001, np.random.uniform() > 0.00001, np.random.uniform() > 0.8]
        elif np.random.uniform() > 0.5 and np.random.uniform() < 0.97:          
            nodes1 = generate_parameters_2()
            nodes2 = nodes1
            flag1 = [np.random.uniform() > 0.00001, np.random.uniform() > 0.00001, np.random.uniform() > 0.00001, np.random.uniform() > 0.8]
        else:                                                                  
            nodes1 = generate_parameters_3()
            flag1 = [np.random.uniform() > 0.00001, np.random.uniform() > 0.00001, np.random.uniform() > 0.9]

        start1 = np.random.uniform(low=0, high=0.3, size=(len(nodes1))).tolist()
        end1 = np.random.uniform(low=0.7, high=1, size=(len(nodes1))).tolist()


        # start/end points of secondary creases     
        n2 = np.random.randint(5, 15)
        coord2 = np.zeros((n2, 2, 2))  
        angle_probs = [(0, np.pi / 6),
                       (5 * np.pi / 6, 7 * np.pi / 6),
                       (11 * np.pi / 6, 2 * np.pi),
                       (np.pi / 3, 2 * np.pi / 3),
                       (4 * np.pi / 3, 5 * np.pi / 3),
                       (0, 2 * np.pi)]
        
        for q in range(n2):
            if np.random.uniform() <= 0.8:
                length = np.random.uniform(50, 100)
            else:
                length = np.random.uniform(100, 150)
            
            angle_prob = np.random.choice([0, 1, 2, 3, 4, 5], p=[0.15, 0.2, 0.15, 0.2, 0.2, 0.1])
            angle_range = angle_probs[angle_prob]
            angle = np.random.uniform(angle_range[0], angle_range[1])
            
            coord2[q, 0] = [np.random.uniform(53, args.imsize - 53), np.random.uniform(53, args.imsize - 53)]
            coord2[q, 1] = [coord2[q, 0, 0] + length * np.cos(angle),
                            coord2[q, 0, 1] + length * np.sin(angle)]  
        
        s2 = np.clip(np.random.normal(scale=0.4, size=(n2,)), -0.15, 0.15)
        t2 = np.clip(np.random.normal(loc=0.5, scale=0.4, size=(n2,)), 0.3, 0.7)


        
        for s in range(args.samples):
            fig = plt.figure(frameon=False)
            canvas = fig.canvas
            dpi = fig.get_dpi()
            fig.set_size_inches((args.imsize + EPS) / dpi, (args.imsize + EPS) / dpi)       
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
            ax = plt.gca()
            ax.set_xlim(0, args.imsize)
            ax.set_ylim(args.imsize, 0)
            ax.axis('off')

            
            img = np.ones((args.imsize, args.imsize), dtype=np.uint8) * 255  
            ax.imshow(img, cmap='gray', vmin=0, vmax=255)  

            
            if nodes2:
                curves1 = [bezier.Curve(n.T * args.imsize + np.random.uniform(-5, 5, size=n.T.shape), degree=3) for n in nodes2]
            else:
                curves1 = [bezier.Curve(n.T * args.imsize + np.random.uniform(-5, 5, size=n.T.shape), degree=2) for n in nodes1]

            points1 = [c.evaluate_multi(np.linspace(s, e, 50)).T for c, s, e in zip(curves1, start1, end1)]          
            paths1 = [Path(p) for p in points1]     
            lw1 = np.random.uniform(2.3, 2.7)       
            patches1 = [patches.PathPatch(p, edgecolor='black', facecolor='none', lw=lw1) for p in paths1]     
            for p, f in zip(patches1, flag1):
                if f:
                    ax.add_patch(p)

            
            coord2_ = coord2 + np.random.uniform(-5, 5, coord2.shape)
            s2_ = s2 + np.random.uniform(-0.1, 0.1, s2.shape)
            t2_ = t2 + np.random.uniform(-0.05, 0.05, s2.shape)
            lw2 = np.random.uniform(0.7, 1.0)       
            for j in range(n2):                     
                points2 = get_bezier(coord2_[j, 0], coord2_[j, 1], t=t2_[j], s=s2_[j]).evaluate_multi(np.linspace(0, 1, 50)).T      
                p = patches.PathPatch(Path(points2), edgecolor='black', facecolor='none', lw=lw2)
                ax.add_patch(p)

            
            stream, _ = canvas.print_to_buffer()
            buffer = np.frombuffer(stream, dtype='uint8')

            
            img = buffer.reshape((args.imsize, args.imsize, 4))  
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)  

            
            _, img_binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY) 

            
            filename = join(args.output, '%.4d' % i, '%.2d.png' % s)            
            os.makedirs(dirname(filename), exist_ok=True)                       
            cv2.imwrite(filename, img_binary)                                          
            plt.close()
            local_idx += 1
        toc = time.time()
        average_meter.update(toc-tic)
        print("proc[%.3d/%.3d] id=%.4d [%.4d/%.4d]  (%.3f sec per id)" % (proc_index, args.nproc, i, id_idx, ids_per_proc, average_meter.avg))  


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)         

    spacing = np.linspace(0, args.num_ids,  args.nproc + 1).astype(int)

    ranges = []
    for i in range(len(spacing) - 1):           
        ranges.append([spacing[i], spacing[i + 1]])

    argins = []
    for p in range(args.nproc):
        argins.append([p, ranges, args])

    with Pool() as pool:
        pool.starmap(batch_process, argins)    

