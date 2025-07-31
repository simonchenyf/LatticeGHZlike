/*

File: subLanczosJ0.c
Author: Yaofeng Chen, Xuanchen Zhang
Organization: Tsinghua University

Description:

    Backend code for Lanczos-based implementation. Used to evolve state to end state
    so as to calculate end state population and parity oscillation in Fig. S1 in the
    Supplementary Information.

License:
MIT License

*/

#define PY_SSIZE_T_CLEAN
#include <stdio.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>
#include <math.h>

void applyJxx(double* tmpr, double* tmpi, const int i, const int N, const Py_complex* v0, const double* J0){
    Py_complex x;
    double coeff;
    for (int j = 0; j < N; j++) {
        for (int k = j+1; k < N; k++){
            x = v0[i ^ ((1<<(N-1-j)) + (1<<(N-1-k)))];
            coeff = J0[j*N+k];
            *tmpr += x.real * coeff;
            *tmpi += x.imag * coeff;
        }
    }
}

void applyJyy(double* tmpr, double* tmpi, const int i, const int N, const Py_complex* v0, const double* J0){
    Py_complex x;
    double coeff;
    int u;
    int spin1, spin2;
    for (int j = 0; j < N; j++) {
        for (int k = j+1; k < N; k++){
            u = 1<<(N-1-k);
            spin2 = (i & u) >> (N-1-k);
            u += 1<<(N-1-j);
            spin1 = (i & u) >> (N-1-j);
            x = v0[i ^ u];
            coeff = pow(-1,spin1) * pow(-1,spin2) * J0[j*N+k];
            *tmpr -= x.real * coeff;
            *tmpi -= x.imag * coeff;
        }
    }
}

static PyObject* py_subLanczosJ0(PyObject* self, PyObject* args) {
    int N;
    double beta;
    PyObject *J0_array_obj;
    PyObject *v0_array_obj;
    PyObject *v1_array_obj;
    PyObject *w_array_obj;
    PyObject *V_array_obj;
    int alongx;

    if (!PyArg_ParseTuple(args, "idOOOOOp", &N, &beta, &J0_array_obj, &v0_array_obj, &v1_array_obj, &w_array_obj, &V_array_obj, &alongx))
        return NULL;

    PyObject *J0_array_cont = PyArray_FROM_OTF(J0_array_obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    PyObject *v0_array_cont = PyArray_FROM_OTF(v0_array_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    PyObject *v1_array_cont = PyArray_FROM_OTF(v1_array_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    PyObject *w_array_cont = PyArray_FROM_OTF(w_array_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    PyObject *V_array_cont = PyArray_FROM_OTF(V_array_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);

    if (J0_array_cont == NULL) {
        return NULL;
    }

    if (v0_array_cont == NULL) {
        return NULL;
    }

    if (v1_array_cont == NULL) {
        return NULL;
    }

    if (w_array_cont == NULL) {
        return NULL;
    }

    if (V_array_cont == NULL) {
        return NULL;
    }

    double *J0_array = (double*)PyArray_DATA(J0_array_cont);
    Py_complex *v0_array = (Py_complex*)PyArray_DATA(v0_array_cont);
    Py_complex *v1_array = (Py_complex*)PyArray_DATA(v1_array_cont);
    Py_complex *w_array = (Py_complex*)PyArray_DATA(w_array_cont);
    Py_complex *V_array = (Py_complex*)PyArray_DATA(V_array_cont);

    void (*applyJ)(double *tmpr, double *tmpi, const int i, const int N, const Py_complex *v0, const double *J0);
    applyJ = alongx ? applyJxx : applyJyy ;

    double alpha = 0.;

    Py_BEGIN_ALLOW_THREADS

    #pragma omp parallel for
    for (int i = 0; i < (1 << N); i++) {
        Py_complex tmp;
        
        tmp = v0_array[i];
        tmp.real /= beta;
        tmp.imag /= beta;
        v0_array[i] = tmp;

	}

    #pragma omp barrier

	#pragma omp parallel for reduction(+:alpha)    
    for (int i = 0; i < (1 << N); i++){
        double tmpr = 0, tmpi = 0;
        Py_complex x;
        applyJ(&tmpr, &tmpi, i, N, v0_array, J0_array);
        
        x = v0_array[i];
        V_array[i] = x;

        alpha += tmpr * x.real + tmpi * x.imag;

        x = v1_array[i];

        w_array[i].real = tmpr - x.real * beta;
        w_array[i].imag = tmpi - x.imag * beta;

    }

    #pragma omp barrier

    #pragma omp parallel for
    for (int i = 0; i < (1 << N); i++){

        Py_complex tmp = v0_array[i];

        tmp.real *= alpha;
        tmp.imag *= alpha;

        w_array[i].real -= tmp.real;
        w_array[i].imag -= tmp.imag;


    }

    Py_END_ALLOW_THREADS

    Py_DECREF(J0_array_cont);
    Py_DECREF(v0_array_cont);
    Py_DECREF(v1_array_cont);
    Py_DECREF(w_array_cont);
    Py_DECREF(V_array_cont);

    return PyFloat_FromDouble(alpha);
}


static PyMethodDef subLanczosJ0_method[] = {
    {"apply", py_subLanczosJ0, METH_VARARGS, "Apply a sub-part of a Lanczos loop with J0."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef subLanczosJ0_module = {
    PyModuleDef_HEAD_INIT,
    "subLanczosJ0",
    NULL,
    -1,
    subLanczosJ0_method
};

PyMODINIT_FUNC PyInit_subLanczosJ0(void) {
    import_array();
    return PyModule_Create(&subLanczosJ0_module);
}
