/*

File: getPred.c
Author: Yaofeng Chen, Xuanchen Zhang
Organization: Tsinghua University

Description:

    Backend code for Lanczos-based implementation. Used to get physical
    quantites (predictions) from Lanczos-evolved state vectors. Used to
    calculate end state population and parity oscillation in Fig. S1 in
    the Supplementary Information.

License:
MIT License

*/

#define PY_SSIZE_T_CLEAN
#include <stdio.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>

static PyObject* py_getPred(PyObject* self, PyObject* args) {
    int N;
    PyObject *psi_obj;
    PyObject *Sz_obj;
    PyObject *SxPsi_obj;
    PyObject *SyPsi_obj;
    PyObject *out_obj;

    if (!PyArg_ParseTuple(args, "iOOOOO", &N, &psi_obj, &Sz_obj, &SxPsi_obj, &SyPsi_obj, &out_obj))
        return NULL;

    PyObject *psi_cont = PyArray_FROM_OTF(psi_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    PyObject *Sz_cont = PyArray_FROM_OTF(Sz_obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    PyObject *SxPsi_cont = PyArray_FROM_OTF(SxPsi_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    PyObject *SyPsi_cont = PyArray_FROM_OTF(SyPsi_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    PyObject *out_cont = PyArray_FROM_OTF(out_obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);

    if (psi_cont == NULL) {
        return NULL;
    }
    
    if (Sz_cont == NULL) {
        return NULL;
    }
    
    if (SxPsi_cont == NULL) {
        return NULL;
    }
    
    if (SyPsi_cont == NULL) {
        return NULL;
    }

    if (out_cont == NULL) {
        return NULL;
    }    

    Py_complex *psi = (Py_complex*)PyArray_DATA(psi_cont);
    double *Sz = (double*)PyArray_DATA(Sz_cont);
    Py_complex *SxPsi = (Py_complex*)PyArray_DATA(SxPsi_cont);
    Py_complex *SyPsi = (Py_complex*)PyArray_DATA(SyPsi_cont);
    double *out = (double*)PyArray_DATA(out_cont);

    double sx=0, sy=0, sz=0, sx2=0, sy2=0, sz2=0, sxsy=0, sysz=0, szsx=0;

    Py_BEGIN_ALLOW_THREADS

    #pragma omp parallel for
    for (int i = 0; i < (1 << N); i++) {
        double tmpxr=0, tmpxi=0, tmpyr=0, tmpyi=0;
        Py_complex x;
        int k;
        int spin;
        for (int j = 0; j < N; j++) {
            k = 1 << j;
            spin = (i & k) >> j;
            // should be 1 << (N-1-j) since leftmost is highest, 
            // but since here coefficients for all sites are equal, can spare the effort
            x = psi[i ^ k];
            tmpxr += x.real;
            tmpxi += x.imag;
            tmpyr += pow(-1,spin) * x.imag;
            tmpyi -= pow(-1,spin) * x.real;
        }
        SxPsi[i].real = tmpxr;
        SxPsi[i].imag = tmpxi;
        SyPsi[i].real = tmpyr;
        SyPsi[i].imag = tmpyi;
    }

    #pragma omp parallel for \
            reduction(+:sx) reduction(+:sy) reduction(+:sz) \
            reduction(+:sx2) reduction(+:sy2) reduction(+:sz2) \
            reduction(+:sxsy) reduction(+:sysz) reduction(+:szsx)
    for (int i = 0; i < (1 << N); i++) {
        Py_complex psi_ele = psi[i];
        Py_complex SxPsi_ele = SxPsi[i];
        Py_complex SyPsi_ele = SyPsi[i];
        double Sz_ele = Sz[i];
        double SzPsir = Sz_ele * psi_ele.real;
        double SzPsii = Sz_ele * psi_ele.imag;

        sx += (psi_ele.real * SxPsi_ele.real + psi_ele.imag * SxPsi_ele.imag);
        sy += (psi_ele.real * SyPsi_ele.real + psi_ele.imag * SyPsi_ele.imag);
        sz += (psi_ele.real * SzPsir + psi_ele.imag * SzPsii);
        sx2 += (SxPsi_ele.real * SxPsi_ele.real + SxPsi_ele.imag * SxPsi_ele.imag);
        sy2 += (SyPsi_ele.real * SyPsi_ele.real + SyPsi_ele.imag * SyPsi_ele.imag);
        sz2 += (SzPsir * SzPsir + SzPsii * SzPsii);
        sxsy += (SxPsi_ele.real * SyPsi_ele.real + SxPsi_ele.imag * SyPsi_ele.imag);
        sysz += (SyPsi_ele.real * SzPsir + SyPsi_ele.imag * SzPsii);
        szsx += (SzPsir * SxPsi_ele.real + SzPsii * SxPsi_ele.imag);
    }

    Py_END_ALLOW_THREADS

    out[0] = sx / 2;
    out[1] = sy / 2;
    out[2] = sz / 2;
    out[3] = sx2 / 4;
    out[4] = sy2 / 4;
    out[5] = sz2 / 4;
    out[6] = sxsy / 4;
    out[7] = sysz / 4;
    out[8] = szsx / 4;

    Py_DECREF(psi_cont);
    Py_DECREF(Sz_cont);
    Py_DECREF(SxPsi_cont);
    Py_DECREF(SyPsi_cont);

    Py_RETURN_NONE;

}


static PyMethodDef getPred_method[] = {
    {"apply", py_getPred, METH_VARARGS, "Get the predictions."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef getPred_module = {
    PyModuleDef_HEAD_INIT,
    "getPred",
    NULL,
    -1,
    getPred_method
};

PyMODINIT_FUNC PyInit_getPred(void) {
    import_array();
    return PyModule_Create(&getPred_module);
}
