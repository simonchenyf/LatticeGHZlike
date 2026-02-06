/*

File: subLanczos.c
Author: Yaofeng Chen, Xuanchen Zhang
Organization: Tsinghua University

Description:

    Backend code for Lanczos-based implementation. Used to calculate end state population
    and parity oscillation in Fig. S1 in the Supplementary Information.

License:
MIT License

*/

#define PY_SSIZE_T_CLEAN
#include <stdio.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>
#include <math.h>

void applySx(double* tmpr, double* tmpi, const int i, const int N, const Py_complex* v0){
    Py_complex x;
    for (int j = 0; j < N; j++) {
        x = v0[i ^ (1 << j)]; 
        // should be 1 << (N-1-j) since leftmost is highest, 
        // but since here coefficients for all sites are equal, can spare the effort
        *tmpr += x.real;
        *tmpi += x.imag;
    }
}

void applySy(double* tmpr, double* tmpi, const int i, const int N, const Py_complex* v0){
    int k;
    int spin;
    Py_complex x;
    for (int j = 0; j < N; j++) {
        k = 1 << j;
        spin = (i & k) >> j;
        // should be (N-1-j) instead of j since leftmost is highest, 
        // but since here coefficients for all sites are equal, can spare the effort
        x = v0[i ^ k];
        *tmpr += pow(-1,spin) * x.imag;
        *tmpi -= pow(-1,spin) * x.real;
    }
}


static PyObject* py_subLanczos(PyObject* self, PyObject* args) {
    int N;
    double beta;
    PyObject *v0_array_obj;
    PyObject *v1_array_obj;
    PyObject *w_array_obj;
    PyObject *V_array_obj;
    int alongx;

    if (!PyArg_ParseTuple(args, "idOOOOp", &N, &beta, &v0_array_obj, &v1_array_obj, &w_array_obj, &V_array_obj, &alongx))
        return NULL;

    PyObject *v0_array_cont = PyArray_FROM_OTF(v0_array_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    PyObject *v1_array_cont = PyArray_FROM_OTF(v1_array_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    PyObject *w_array_cont = PyArray_FROM_OTF(w_array_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    PyObject *V_array_cont = PyArray_FROM_OTF(V_array_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);

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

    Py_complex *v0_array = (Py_complex*)PyArray_DATA(v0_array_cont);
    Py_complex *v1_array = (Py_complex*)PyArray_DATA(v1_array_cont);
    Py_complex *w_array = (Py_complex*)PyArray_DATA(w_array_cont);
    Py_complex *V_array = (Py_complex*)PyArray_DATA(V_array_cont);

    void (*applyS)(double *tmpr, double *tmpi, const int i, const int N, const Py_complex *v0);
    applyS = alongx ? applySx : applySy ;

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
        applyS(&tmpr, &tmpi, i, N, v0_array);
        
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

    Py_DECREF(v0_array_cont);
    Py_DECREF(v1_array_cont);
    Py_DECREF(w_array_cont);
    Py_DECREF(V_array_cont);

    return PyFloat_FromDouble(alpha);
}


static PyMethodDef subLanczos_method[] = {
    {"apply", py_subLanczos, METH_VARARGS, "Apply a sub-part of a Lanczos loop."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef subLanczos_module = {
    PyModuleDef_HEAD_INIT,
    "subLanczos",
    NULL,
    -1,
    subLanczos_method
};

PyMODINIT_FUNC PyInit_subLanczos(void) {
    import_array();
    return PyModule_Create(&subLanczos_module);
}
