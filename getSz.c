/*

File: getSz.c
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

static PyObject* py_getSz(PyObject* self, PyObject* args) {
    int N;
    PyObject *array_obj;

    if (!PyArg_ParseTuple(args, "iO", &N, &array_obj))
        return NULL;

    PyObject *array_cont = PyArray_FROM_OTF(array_obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);

    if (array_cont == NULL) {
        return NULL;
    }

    double *array = (double*)PyArray_DATA(array_cont);

    Py_BEGIN_ALLOW_THREADS

    #pragma omp parallel for
    for (int i = 0; i < (1 << N); i++) {
        array[i] = N - 2 * __builtin_popcount(i);
    }

    Py_END_ALLOW_THREADS

    Py_DECREF(array_cont);

    Py_RETURN_NONE;
}


static PyMethodDef getSz_method[] = {
    {"apply", py_getSz, METH_VARARGS, "Get the diagonal of Sz."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef getSz_module = {
    PyModuleDef_HEAD_INIT,
    "getSz",
    NULL,
    -1,
    getSz_method
};

PyMODINIT_FUNC PyInit_getSz(void) {
    import_array();
    return PyModule_Create(&getSz_module);
}