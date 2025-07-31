/*

File: zeroize.c
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

static PyObject* py_zeroize(PyObject* self, PyObject* args) {
    int N;
    int m;
    PyObject *V_array_obj;
    PyObject *v1_array_obj;
    PyObject *w_array_obj;

    if (!PyArg_ParseTuple(args, "iiOOO", &N, &m, &V_array_obj, &v1_array_obj, &w_array_obj))
        return NULL;

    PyObject *V_array_cont = PyArray_FROM_OTF(V_array_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    PyObject *v1_array_cont = PyArray_FROM_OTF(v1_array_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    PyObject *w_array_cont = PyArray_FROM_OTF(w_array_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);

    if (V_array_cont == NULL) {
        return NULL;
    }

    if (v1_array_cont == NULL) {
        return NULL;
    }
    
    if (w_array_cont == NULL) {
        return NULL;
    }

    Py_complex *V_array = (Py_complex*)PyArray_DATA(V_array_cont);
    Py_complex *v1_array = (Py_complex*)PyArray_DATA(v1_array_cont);
    Py_complex *w_array = (Py_complex*)PyArray_DATA(w_array_cont);

    Py_BEGIN_ALLOW_THREADS

    #pragma omp parallel for
    for (int i = 0; i < (1 << N); i++) {
        v1_array[i].real = 0;
        v1_array[i].imag = 0;
        w_array[i].real = 0;
        w_array[i].imag = 0;
        long offset = m * i;
        for (int j = 0; j < m; j++) {
            V_array[offset + j].real = 0; 
            V_array[offset + j].imag = 0; 
        }
    }

    Py_END_ALLOW_THREADS

    Py_DECREF(V_array_cont);
    Py_DECREF(v1_array_cont);
    Py_DECREF(w_array_cont);

    Py_RETURN_NONE;
}


static PyMethodDef zeroize_method[] = {
    {"apply", py_zeroize, METH_VARARGS, "Zero-ize all input arrays."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef zeroize_module = {
    PyModuleDef_HEAD_INIT,
    "zeroize",
    NULL,
    -1,
    zeroize_method
};

PyMODINIT_FUNC PyInit_zeroize(void) {
    import_array();
    return PyModule_Create(&zeroize_module);
}