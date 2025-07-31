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
import random

from multiprocessing import Pool

from vlkit.image import norm255
from vlkit.utils import AverageMeter

import os, sys, argparse, glob, cv2, random, time
from os.path import join, split, isdir, isfile, dirname
import shutil

from copy import copy
from tqdm import tqdm
from PIL import Image, ImageFilter, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args():
    parser = argparse.ArgumentParser(description='Compare results')     
    #添加命令行参数    #default为默认值
    parser.add_argument('--num_ids', type=int, default=4000)             
    parser.add_argument('--samples', type=int, default=100)              
    parser.add_argument('--nproc', type=int, default=100)                 
    parser.add_argument('--imsize', type=int, default=256)              
    parser.add_argument('--output', type=str, default='./pce-bezier-40w')    
    args = parser.parse_args()
    assert args.num_ids % args.nproc == 0   
    return args


def sample_edge(low, high):
    """
    sample points on edges of a unit square 将样本点限制在单位正方形边缘
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
    v = v / max(np.sqrt((v ** 2).sum()), 1e-6)     
    return c + s * l * v                        


def get_bezier(p0, p1, t=0.5, s=1):
    assert -1 < s < 1, 's=%f' % s             
    c = control_point(p0, p1, t, s)
    nodes = np.vstack((p0, c, p1)).T
    return bezier.Curve(nodes, degree=2)


def generate_parameters_norm():
    # head coordinates
    # head1  =  np.array([np.random.uniform(low=0.1, high=0.3), np.random.uniform(low=0, high=0.05)])
    # head2  =  np.array([np.random.uniform(low=0.15, high=0.3), np.random.uniform(low=0, high=0.05)])
    ## [x, y]
    head1 = np.array([np.random.uniform(low=0.2, high=0.6), np.random.uniform(low=0.05, high=0.15)])
    head2 = np.array([np.random.uniform(low=0.0, high=0.05), np.random.uniform(low=0.25, high=0.3)])
    head3 = np.array([np.random.uniform(low=0.0, high=0.5), np.random.uniform(low=0.25, high=0.35)])
    head4 = np.array([0, 0])

    # tail coordinates
    tail1 = np.array([np.random.uniform(low=0.99, high=1), np.random.uniform(low=0.25, high=0.3)])
    tail2 = np.array([np.random.uniform(low=0.5, high=0.8), np.random.uniform(low=0.25, high=0.3)])
    tail3 = np.array([np.random.uniform(low=0.4, high=0.5), np.random.uniform(low=0.9, high=1)])
    tail4 = np.array([0, 0])

    c1 = control_point(head1, tail1, s=-np.random.uniform(0.05, 0.1))
    c2 = control_point(head2, tail2, s=np.random.uniform(0.05, 0.1))
    c3 = control_point(head3, tail3, s=np.random.uniform(0.1, 0.2))
    c4 = control_point(head4, tail4, s=-np.random.uniform(0.1, 0.12))

    return np.vstack((head1, c1, tail1)), np.vstack((head2, c2, tail2)), np.vstack((head3, c3, tail3)), np.vstack(
        (head4, c4, tail4))


def generate_parameters_cross():
    # head coordinates
    # head1  =  np.array([np.random.uniform(low=0.1, high=0.3), np.random.uniform(low=0, high=0.05)])
    # head2  =  np.array([np.random.uniform(low=0.15, high=0.3), np.random.uniform(low=0, high=0.05)])

    ## 掌纹起始/结束点坐标[x, y]
    head1 = np.array([np.random.uniform(low=0.0, high=0.05), np.random.uniform(low=0.25, high=0.3)])
    head2 = np.array([np.random.uniform(low=0.0, high=0.5), np.random.uniform(low=0.25, high=0.35)])
    head3 = np.array([0, 0])
    head4 = np.array([0, 0])

    # tail coordinates
    tail1 = np.array([np.random.uniform(low=0.99, high=1), np.random.uniform(low=0.25, high=0.3)])
    tail2 = np.array([np.random.uniform(low=0.4, high=0.5), np.random.uniform(low=0.9, high=1)])
    tail3 = np.array([0, 0])
    tail4 = np.array([0, 0])

    c1 = control_point(head1, tail1, s=np.random.uniform(0.05, 0.1))
    c2 = control_point(head2, tail2, s=np.random.uniform(0.1, 0.2))
    c3 = control_point(head3, tail3, s=-np.random.uniform(0.1, 0.12))
    c4 = control_point(head4, tail4, s=-np.random.uniform(0.1, 0.12))

    return np.vstack((head1, c1, tail1)), np.vstack((head2, c2, tail2)), np.vstack((head3, c3, tail3)), np.vstack(
        (head4, c4, tail4))


def generate_parameters_bridge():
    head1 = np.array([np.random.uniform(low=0.2, high=0.6), np.random.uniform(low=0.05, high=0.15)])
    head2 = np.array([np.random.uniform(low=0.0, high=0.05), np.random.uniform(low=0.25, high=0.3)])
    head3 = np.array([np.random.uniform(low=0.0, high=0.5), np.random.uniform(low=0.25, high=0.35)])
    head4 = np.array([np.random.uniform(low=0.5, high=0.6), np.random.uniform(low=0.05, high=0.15)])

    # tail coordinates
    tail1 = np.array([np.random.uniform(low=0.99, high=1), np.random.uniform(low=0.25, high=0.3)])
    tail2 = np.array([np.random.uniform(low=0.6, high=1), np.random.uniform(low=0.25, high=0.3)])
    tail3 = np.array([np.random.uniform(low=0.4, high=0.5), np.random.uniform(low=0.9, high=1)])
    tail4 = np.array([np.random.uniform(low=0.4, high=0.5), np.random.uniform(low=0.2, high=0.4)])

    c1 = control_point(head1, tail1, s=-np.random.uniform(0.05, 0.1))
    c2 = control_point(head2, tail2, s=np.random.uniform(0.05, 0.1))
    c3 = control_point(head3, tail3, s=np.random.uniform(0.1, 0.2))
    c4 = control_point(head4, tail4, s=-np.random.uniform(0.1, 0.12))

    return np.vstack((head1, c1, tail1)), np.vstack((head2, c2, tail2)), np.vstack((head3, c3, tail3)), np.vstack(
        (head4, c4, tail4))


def generate_parameters_fork():
    head1 = np.array([np.random.uniform(low=0.2, high=0.6), np.random.uniform(low=0.05, high=0.15)])
    head2 = np.array([np.random.uniform(low=0.0, high=0.05), np.random.uniform(low=0.25, high=0.3)])
    head3 = np.array([np.random.uniform(low=0.0, high=0.5), np.random.uniform(low=0.25, high=0.35)])
    head4 = np.array([np.random.uniform(low=0.48, high=0.52), np.random.uniform(low=0.28, high=0.32)])

    # tail coordinates
    tail1 = np.array([np.random.uniform(low=0.99, high=1), np.random.uniform(low=0.25, high=0.3)])
    tail2 = np.array([np.random.uniform(low=0.48, high=0.52), np.random.uniform(low=0.25, high=0.3)])
    tail3 = np.array([np.random.uniform(low=0.4, high=0.5), np.random.uniform(low=0.9, high=1)])
    tail4 = np.array([np.random.uniform(low=0.7, high=0.95), np.random.uniform(low=0.25, high=0.3)])

    c1 = control_point(head1, tail1, s=-np.random.uniform(0.05, 0.1))
    c2 = control_point(head2, tail2, s=np.random.uniform(0.05, 0.1))
    c3 = control_point(head3, tail3, s=np.random.uniform(0.1, 0.2))
    c4 = control_point(head4, tail4, s=-np.random.uniform(0.1, 0.12))

    return np.vstack((head1, c1, tail1)), np.vstack((head2, c2, tail2)), np.vstack((head3, c3, tail3)), np.vstack(
        (head4, c4, tail4))


def generate_parameters_sydney():
    # head coordinates
    # head1  =  np.array([np.random.uniform(low=0.1, high=0.3), np.random.uniform(low=0, high=0.05)])
    # head2  =  np.array([np.random.uniform(low=0.15, high=0.3), np.random.uniform(low=0, high=0.05)])
    ## [x, y]
    head1 = np.array([np.random.uniform(low=0.2, high=0.6), np.random.uniform(low=0.05, high=0.15)])
    head2 = np.array([np.random.uniform(low=0.0, high=0.05), np.random.uniform(low=0.25, high=0.3)])
    head3 = np.array([np.random.uniform(low=0.0, high=0.5), np.random.uniform(low=0.25, high=0.35)])
    head4 = np.array([0, 0])

    # tail coordinates
    tail1 = np.array([np.random.uniform(low=0.99, high=1), np.random.uniform(low=0.25, high=0.3)])
    tail2 = np.array([np.random.uniform(low=0.9, high=1), np.random.uniform(low=0.25, high=0.3)])
    tail3 = np.array([np.random.uniform(low=0.4, high=0.5), np.random.uniform(low=0.9, high=1)])
    tail4 = np.array([0, 0])

    c1 = control_point(head1, tail1, s=-np.random.uniform(0.05, 0.1))
    c2 = control_point(head2, tail2, s=np.random.uniform(0.05, 0.1))
    c3 = control_point(head3, tail3, s=np.random.uniform(0.1, 0.2))
    c4 = control_point(head4, tail4, s=-np.random.uniform(0.1, 0.12))

    return np.vstack((head1, c1, tail1)), np.vstack((head2, c2, tail2)), np.vstack((head3, c3, tail3)), np.vstack(
        (head4, c4, tail4))



def batch_process(proc_index, ranges, args):
    ids_per_proc = int(args.num_ids / args.nproc)   
    EPS = 1e-2

    np.random.seed(proc_index)
    random.seed(proc_index)

    average_meter = AverageMeter(name='time')
    start = 0
    end = 4
    local_idx = 0

    for id_idx, i in enumerate(range(*ranges[proc_index])):
        tic = time.time()

        random_integer = random.randint(start, end)

        # generate_norm
        if (random_integer == 0):
            nodes1 = generate_parameters_norm()
            flag1 = [np.random.uniform() > 0.001, np.random.uniform() > 0.001, np.random.uniform() > 0.001,
                     np.random.uniform() > 1]

        # generate_bridge
        elif (random_integer == 1):
            nodes1 = generate_parameters_bridge()
            flag1 = [np.random.uniform() > 0.001, np.random.uniform() > 0.001, np.random.uniform() > 0.001,
                     np.random.uniform() > 1]

        # generate_fork
        elif (random_integer == 2):
            nodes1 = generate_parameters_fork()
            flag1 = [np.random.uniform() > 0.001, np.random.uniform() > 0.001, np.random.uniform() > 0.001,
                     np.random.uniform() > 1]

        # generate_cross
        elif (random_integer == 3):
            nodes1 = generate_parameters_cross()
            flag1 = [np.random.uniform() > 0.001, np.random.uniform() > 0.001, np.random.uniform() > 0.001,
                     np.random.uniform() > 1]

        # generate_sydney
        elif (random_integer == 4):
            nodes1 = generate_parameters_sydney()
            flag1 = [np.random.uniform() > 0.001, np.random.uniform() > 0.001, np.random.uniform() > 0.001,
                     np.random.uniform() > 1]

        start1 = np.random.uniform(low=0, high=0.10, size=(len(nodes1))).tolist()
        end1 = np.random.uniform(low=0.90, high=1, size=(len(nodes1))).tolist()


        # start/end points of secondary creases     
        n2 = np.random.randint(5, 15)               
        coord2 = np.random.uniform(0, args.imsize, size=(n2, 2, 2))         

        for k in range(n2):
            r = np.random.uniform(0.0, 2 * np.pi)    
            delta = np.array([np.cos(r), np.sin(r)]) * np.random.uniform(0.1, 0.5) * args.imsize   
            coord2[k][1] = coord2[k][0] + delta      

        s2 = np.clip(np.random.normal(scale=0.4, size=(n2,)), -0.6, 0.6)
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


            curves1 = [bezier.Curve(n.T * args.imsize, degree=2) for n in nodes1]
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
            lw2 = np.random.uniform(0.9, 1.1)       
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

