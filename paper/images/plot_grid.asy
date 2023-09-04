
// Specifying input file
file fin=input('gspc.txt');

// Reading reciprocal lattice vectors and tpiba
triple recvec_1 = (fin, fin, fin);
triple recvec_2 = (fin, fin, fin);
triple recvec_3 = (fin, fin, fin);
real tpiba = fin;

// Constructing transform3 for
// crystal to cartesian transformations
transform3 recilat = copy(identity4);
recilat[0][0] = recvec_1.x;
recilat[1][0] = recvec_1.y;
recilat[2][0] = recvec_1.z;
recilat[0][1] = recvec_2.x;
recilat[1][1] = recvec_2.y;
recilat[2][1] = recvec_2.z;
recilat[0][2] = recvec_3.x;
recilat[1][2] = recvec_3.y;
recilat[2][2] = recvec_3.z;
transform3 recilat_inv = inverse(recilat);

// Reading cutoff radius and FFT grid shape
real rsphere = fin;
triple boxshape = (fin, fin, fin);

// Reading G-vectors data
int ng = fin;
triple[] gcryst = new triple[ng];
real[] gcol = new real[ng];
for (int i=0; i<ng; ++i) {
    gcryst[i] = (fin, fin, fin);
    gcol[i] = fin;
}

// Reading G-sticks data
int ns = fin;
pair[] scryst = new pair[ns];
real[] scol = new real[ns];
for (int i=0; i<ns; ++i) {
    scryst[i] = (fin, fin);
    scol[i] = fin;
}
//--------------------------------------------------------
// All data read from file; now we can generate the plot

import three;
settings.outformat="pdf";
settings.render=4;
size(10cm,0);

// Overwriting recvec and recilat with unit vectors ()
// for crystal coordinates
real uscale = 1.0 / tpiba;


// ---------- Camera ----------
// For cartesian plot, we would like the sticks to be aligned
// vertically, and the camera to be within the first octant
real cameradist = 10 * uscale;
currentprojection = perspective(
    scale3(cameradist)
    * unit(2*Y + Z + 1*X),
    up=X
);

// ---------- FFT Meshgrid ----------
// Drawing the FFT Box (in cartesian)
draw(box(
    (-0.5 * boxshape.x, -0.5 * boxshape.y, -0.5 * boxshape.z),
    (+0.5 * boxshape.x, +0.5 * boxshape.y, +0.5 * boxshape.z)
    ), black);

// Drawing reciprocal lattice vectors as axes
real axscale = 15. * uscale;  // Length of the axis vectors
real axsize = 0.5 * uscale;  // Thickness of the axis vector lines
Label L1 = Label("$X$", position=EndPoint);
draw(O--scale3(axscale)*X, 
    red+linewidth(axsize), 
    arrow=Arrow3(), L=L1);

Label L2 = Label("$Y$", position=EndPoint);
draw(O--scale3(axscale)*Y,
    green+linewidth(axsize),
    arrow=Arrow3(), L=L2);

Label L3 = Label("$Z$", position=EndPoint);
draw(O--scale3(axscale)*Z,
    blue+linewidth(axsize),
    arrow=Arrow3(), L=L3); 

axscale = axscale * 0.75;
L1 = Label("$b_1$", position=EndPoint);
draw(O--scale3(axscale)*unit(recvec_1), 
    dashed+red+linewidth(axsize), 
    arrow=Arrow3(), L=L1);

L2 = Label("$b_2$", position=EndPoint);
draw(O--scale3(axscale)*unit(recvec_2),
    dashed+green+linewidth(axsize),
    arrow=Arrow3(), L=L2);

L3 = Label("$b_3$", position=EndPoint);
draw(O--scale3(axscale)*unit(recvec_3),
    dashed+blue+linewidth(axsize),
    arrow=Arrow3(), L=L3); 

// ---------- G-vectors ----------
// Plotting G-vectors as colored points
real dotsize = 2.0 * uscale;  // Size of each point

triple point;
real c;
for (int i=0; i<ng; ++i) {
    point = copy(gcryst[i]);
    if (point.x > boxshape.x / 2) {
        point.x = point.x - boxshape.x;
    }
    if (point.y > boxshape.y / 2) {
        point.y = point.y - boxshape.y;
    }
    if (point.z > boxshape.z / 2) {
        point.z = point.z - boxshape.z;
    }
    c = gcol[i];
    // 0 - Red, 1 - Green, 2 - Blue, else - Black
    if (c == 0) {
        dot(point, red+linewidth(dotsize));
    } else if (c == 1) {
        dot(point, green+linewidth(dotsize));
    } else if (c == 2) {
        dot(point, blue+linewidth(dotsize));
    } else {
        dot(point, black+linewidth(dotsize));
    }
}

// ---------- Cutoff Sphere ----------
draw(recilat_inv * scale3(rsphere) * unitsphere,
     surfacepen=white+opacity(0.3 * uscale),
     meshpen=gray(0.4 * uscale));
     
