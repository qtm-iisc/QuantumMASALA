import three;
size(4cm,0);


file fin=input('g_cryst.txt');

bool cart=true;


triple recvec_a = (fin, fin, fin);
triple recvec_b = (fin, fin, fin);
triple recvec_c = (fin, fin, fin);

transform3 recilat;
if (cart == true) {
    recilat = copy(identity4);
    recilat[0][0] = recvec_a.x;
    recilat[1][0] = recvec_a.y;
    recilat[2][0] = recvec_a.z;
    recilat[0][1] = recvec_b.x;
    recilat[1][1] = recvec_b.y;
    recilat[2][1] = recvec_b.z;
    recilat[0][2] = recvec_c.x;
    recilat[1][2] = recvec_c.y;
    recilat[2][2] = recvec_c.z;
} else {
    recilat = copy(identity4);
}


// Plotting Basis vectors of reciprocal lattice
real axscale = 5.;
real axsize = 1.5;

triple ax1, ax2, ax3;
if (cart == true) {
    ax1 = recvec_a;
    ax2 = recvec_b;
    ax3 = recvec_c;
} else {
    ax1 = (1, 0, 0);
    ax2 = (0, 1, 0);
    ax3 = (0, 0, 1);
}


// Drawing a FFT Grid box
real rsphere = fin;
triple boxshape = (fin, fin, fin);

draw(recilat * box((-0.5 * boxshape.x, -0.5 * boxshape.y, -0.5 * boxshape.z),
                   (+0.5 * boxshape.x, +0.5 * boxshape.y, +0.5 * boxshape.z)),
     black);
         
triple axshift = (-0.5*boxshape.x, -0.5*boxshape.y, -0.5*boxshape.z);
axshift = recilat * axshift;
Label Lx = Label("$x$", position=EndPoint);
Label Ly = Label("$y$", position=EndPoint);
Label Lz = Label("$z$", position=EndPoint);
draw(axshift--shift(axshift)*scale3(axscale)*ax1, 
    red+linewidth(axsize), arrow=Arrow3(), L=Lx);
draw(axshift--shift(axshift)*scale3(axscale)*ax2,
    green+linewidth(axsize), arrow=Arrow3(), L=Ly);
draw(axshift--shift(axshift)*scale3(axscale)*ax3,
    blue+linewidth(axsize), arrow=Arrow3(), L=Lz); 


triple point;
int c, i, n;


// Plotting G-vector points
real dotsize=0.7;
n = fin;
for (i=0; i<n; ++i) {
    point = recilat * (fin, fin, fin);
    c = fin;
    if (c == 0) {
    dot(point, red+linewidth(dotsize));
    }
    if (c == 1) {
    dot(point, green+linewidth(dotsize));
    }
    if (c == 2) {
    dot(point, blue+linewidth(dotsize));
    }
}

if (cart == true) {
    draw(scale3(rsphere) * unitsphere,
        surfacepen=white+opacity(0.3),
        meshpen=gray(0.4));
}

// Plotting Sticks
real sticksize=0.1;

triple line_start, line_end;
n = fin;
for (i=0; i<n; ++i) {
    point = (0.5 * boxshape.x, fin, fin);
    c = fin;
    line_start = recilat * (-point.x, point.y, point.z);
    line_end   = recilat * (+point.x, point.y, point.z);
    
    if (c == 0) {
    draw(line_start--line_end, red+linewidth(sticksize));
    }
    if (c == 1) {
    draw(line_start--line_end, green+linewidth(sticksize));
    }
    if (c == 2) {
    draw(line_start--line_end, blue+linewidth(sticksize));
    }
}


// axshift = (0, -0.5*boxshape.y, -0.5*boxshape.z);
// for (i=0; i<boxshape.x; ++i) {
//     axshift = (i - 0.5*boxshape.x, -0.5*boxshape.y, -0.5*boxshape.z);
//     draw(recilat * shift(axshift) * scale(boxshape.x, boxshape.y, boxshape.z)
//         * reflect(O, Y, X+Z) * unitplane, 
//         //meshpen=nullpen,
//         surfacepen=red+opacity(0.9));
// }
