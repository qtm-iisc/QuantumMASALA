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


currentprojection = perspective(2*recvec_b+recvec_c+0.5*recvec_a, up=recvec_a);
// Plotting Basis vectors of reciprocal lattice
real axscale = 9.;
real axsize = 0.5;

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
         
triple axshift = (0, 0, 0);
axshift = recilat * axshift;

Label Lx = Label("$a$", position=EndPoint);
Label Ly = Label("$b$", position=EndPoint);
Label Lz = Label("$c$", position=EndPoint);
draw(axshift--shift(axshift)*scale3(axscale)*ax1, 
    red+linewidth(axsize), 
    arrow=Arrow3(), L=Lx);
draw(axshift--shift(axshift)*scale3(axscale)*ax2,
    green+linewidth(axsize),
    arrow=Arrow3(), L=Ly);
draw(axshift--shift(axshift)*scale3(axscale)*ax3,
    blue+linewidth(axsize),
    arrow=Arrow3(), L=Lz); 


triple point;
int c, i, n;


// Plotting G-vector points
real dotsize=1.0;
n = fin;

triple[] points = new triple[n];
for (i=0; i<n; ++i) {
    point = recilat * (fin, fin, fin);
    points[i] = point;
    c = fin;
    c = i % 3;
    if (c == 0) {
    dot(point, red+linewidth(dotsize));
    }
    else if (c == 1) {
    dot(point, green+linewidth(dotsize));
    }
    else if (c == 2) {
    dot(point, blue+linewidth(dotsize));
    }
    else {
    dot(point, black+linewidth(dotsize));
    }
}

if (cart == true) {
    draw(scale3(rsphere) * unitsphere,
        surfacepen=white+opacity(0.3),
        meshpen=gray(0.4));
}
