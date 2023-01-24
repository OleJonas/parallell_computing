/*
 * GENANN - Minimal C Artificial Neural Network
 *
 * Copyright (c) 2015-2018 Lewis Van Winkle
 *
 * http://CodePlea.com
 *
 * This software is provided 'as-is', without any express or implied
 * warranty. In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgement in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 *
 */

#include "genann.h"

#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cblas.h>

#ifndef genann_act
#define genann_act_hidden genann_act_hidden_indirect
#define genann_act_output genann_act_output_indirect
#else
#define genann_act_hidden genann_act
#define genann_act_output genann_act
#endif

#define LOOKUP_SIZE 4096

double genann_act_hidden_indirect(const struct genann *ann, double a) {
    return ann->activation_hidden(ann, a);
}

double genann_act_output_indirect(const struct genann *ann, double a) {
    return ann->activation_output(ann, a);
}

const double sigmoid_dom_min = -15.0;
const double sigmoid_dom_max = 15.0;
double interval;
double lookup[LOOKUP_SIZE];

#ifdef __GNUC__
#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)
#define unused          __attribute__((unused))
#else
#define likely(x)       x
#define unlikely(x)     x
#define unused
#pragma warning(disable : 4996) /* For fscanf */
#endif


double genann_act_sigmoid(const genann *ann unused, double a) {
    if (a < -45.0) return 0;
    if (a > 45.0) return 1;
    return 1.0 / (1 + exp(-a));
}

void genann_init_sigmoid_lookup(const genann *ann) {
        const double f = (sigmoid_dom_max - sigmoid_dom_min) / LOOKUP_SIZE;
        int i;

        interval = LOOKUP_SIZE / (sigmoid_dom_max - sigmoid_dom_min);
        for (i = 0; i < LOOKUP_SIZE; ++i) {
            lookup[i] = genann_act_sigmoid(ann, sigmoid_dom_min + f * i);
        }
}

double genann_act_sigmoid_cached(const genann *ann unused, double a) {
    assert(!isnan(a));

    if (a < sigmoid_dom_min) return lookup[0];
    if (a >= sigmoid_dom_max) return lookup[LOOKUP_SIZE - 1];

    size_t j = (size_t)((a-sigmoid_dom_min)*interval+0.5);

    /* Because doubleing point... */
    if (unlikely(j >= LOOKUP_SIZE)) return lookup[LOOKUP_SIZE - 1];

    return lookup[j];
}

double genann_act_linear(const struct genann *ann unused, double a) {
    return a;
}

double genann_act_threshold(const struct genann *ann unused, double a) {
    return a > 0;
}

genann *genann_init(int inputs, int hidden_layers, int hidden, int outputs) {
    if (hidden_layers < 0) return 0;
    if (inputs < 1) return 0;
    if (outputs < 1) return 0;
    if (hidden_layers > 0 && hidden < 1) return 0;


    const int hidden_weights = hidden_layers ? (inputs+1) * hidden + (hidden_layers-1) * (hidden+1) * hidden : 0;
    const int output_weights = (hidden_layers ? (hidden+1) : (inputs+1)) * outputs;
    const int total_weights = (hidden_weights + output_weights);

    const int total_neurons = (inputs + hidden * hidden_layers + outputs);

    /* Allocate extra size for weights, outputs, and deltas. */
    const int size = sizeof(genann) + sizeof(double) * (total_weights + total_neurons + (total_neurons - inputs));
    genann *ret = malloc(size);
    if (!ret) return 0;

    ret->inputs = inputs;
    ret->hidden_layers = hidden_layers;
    ret->hidden = hidden;
    ret->outputs = outputs;

    ret->total_weights = total_weights;
    ret->total_neurons = total_neurons;

    /* Set pointers. */
    ret->weight = (double*)((char*)ret + sizeof(genann));
    ret->output = ret->weight + ret->total_weights;
    ret->delta = ret->output + ret->total_neurons;

    genann_randomize(ret);

    ret->activation_hidden = genann_act_sigmoid_cached;
    ret->activation_output = genann_act_sigmoid_cached;

    genann_init_sigmoid_lookup(ret);

    return ret;
}


genann *genann_read(FILE *in) {
    int inputs, hidden_layers, hidden, outputs;
    int rc;

    errno = 0;
    rc = fscanf(in, "%d %d %d %d", &inputs, &hidden_layers, &hidden, &outputs);
    if (rc < 4 || errno != 0) {
        perror("fscanf");
        return NULL;
    }

    genann *ann = genann_init(inputs, hidden_layers, hidden, outputs);

    int i;
    for (i = 0; i < ann->total_weights; ++i) {
        errno = 0;
        rc = fscanf(in, " %le", ann->weight + i);
        if (rc < 1 || errno != 0) {
            perror("fscanf");
            genann_free(ann);

            return NULL;
        }
    }

    return ann;
}


genann *genann_copy(genann const *ann) {
    const int size = sizeof(genann) + sizeof(double) * (ann->total_weights + ann->total_neurons + (ann->total_neurons - ann->inputs));
    genann *ret = malloc(size);
    if (!ret) return 0;

    memcpy(ret, ann, size);

    /* Set pointers. */
    ret->weight = (double*)((char*)ret + sizeof(genann));
    ret->output = ret->weight + ret->total_weights;
    ret->delta = ret->output + ret->total_neurons;

    return ret;
}


void genann_randomize(genann *ann) {
    int i;
    for (i = 0; i < ann->total_weights; ++i) {
        double r = GENANN_RANDOM();
        /* Sets weights from -0.5 to 0.5. */
        ann->weight[i] = r - 0.5;
    }
}


void genann_free(genann *ann) {
    /* The weight, output, and delta pointers go to the same buffer. */
    free(ann);
}

double const *genann_run(genann const *ann, double const *inputs) {
    //Copy the weights into a more convenient variable
    double const *w = ann->weight;
    //Initialize the output vector to point at the second layer of neurons
    double *o = ann->output + ann->inputs;
    //Initialize the input vector to point at the first layer of neurons
    double const *i = ann->output;

    /* Copy the inputs to the scratch area, where we also store each neuron's
     * output, for consistency. This way the first layer isn't a special case. */
    memcpy(ann->output, inputs, sizeof(double) * ann->inputs);

    int h, j, k;

    if (!ann->hidden_layers) {
        double *ret = o;
        for (j = 0; j < ann->outputs; ++j) {
            double sum = *w++ * -1.0;
            for (k = 0; k < ann->inputs; ++k) {
                sum += *w++ * i[k];
            }
            *o++ = genann_act_output(ann, sum);
        }

        return ret;
    }

    /* Figure input layer */
    for (j = 0; j < ann->hidden; ++j) {
        double sum = *w++ * -1.0;
        for (k = 0; k < ann->inputs; ++k) {
            sum += *w++ * i[k];
        }
        *o++ = genann_act_hidden(ann, sum);
    }

    i += ann->inputs;

    /////////////////////////////////
    // TODO: 1 GEMV using BLAS   //
    /////////////////////////////////


    /*
        Function description:

        Activates each neuron in the input layer based on an input vector.
        Propagates these activations through the network.

        Variables:
        ann:    The neural network
                ann->weights: The edge and neuron weights for all layers
                ann->inputs: The number of neurons in the input layer
                ann->hidden: The number of neurons in the hidden layers
                ann->hidden_layers: The number of hidden layers in the network
                ann->output: The neuron activations for the whole network, incluing input, hidden and final layer.
        inputs: The input vector
        w:      Points at ann->weights
        o:      Points at the layer of ann->output we are currently writing to
        i:      Points to the layer of ann->output we are currently using as input.
                i.e. the previous layer.

        When the function returns, ann->output contains the final result.
    */


    /* Comment from original source: Figure hidden layers, if any. */

    //These are the dimensions of the square weight matrix
    /* int m = ann->hidden;
    int n = ann->hidden;

    //In addition to n edge weights, each neuron has one value (bias) associated with it.
    //This value is *also* saved in w, meaning the complete matrix has (n+1)*m elements.
    //This value is always multiplied by -1, which we make room for in a copy of the input vector.
    double* temp_i = malloc( (ann->hidden+1) * sizeof(double) );
    double* sums = calloc((ann->hidden+1),  sizeof(double));

    for (h = 1; h < ann->hidden_layers; ++h) {
        //Copyyng the input vector and setting the first value to -1 as described above.
        temp_i[0] = -1.0;
        memcpy(temp_i+1, i, n*sizeof(double));

        ////////////////////////////////////////////////////////////
        // Decompose and replace this double for loop with GEMV call
        // HOTSPOT Part 1: ~19%
        for (j = 0; j < ann->hidden; ++j) {
            for (k = 0; k < ann->hidden+1; ++k) {
                sums[j] += w[k + j*(ann->hidden+1)] * temp_i[k];
            }
            o[j] = genann_act_hidden(ann, sums[j]);
        }
        ////////////////////////////////////////////////////////////

        w += (n + 1) * m;
        o += m;
        i += m;
    }
    free(temp_i);
    free(sums); */

    //These are the dimensions of the square weight matrix
    int m = ann->hidden;
    int n = ann->hidden;

    //In addition to n edge weights, each neuron has one value (bias) associated with it.
    //This value is *also* saved in w, meaning the complete matrix has (n+1)*m elements.
    //This value is always multiplied by -1, which we make room for in a copy of the input vector.
    double* temp_i = malloc( (ann->hidden+1) * sizeof(double) );
    double* sums = calloc((ann->hidden+1),  sizeof(double));

    for (h = 1; h < ann->hidden_layers; ++h) {
        //Copyyng the input vector and setting the first value to -1 as described above.
        temp_i[0] = -1.0;
        memcpy(temp_i+1, i, n*sizeof(double));

        ////////////////////////////////////////////////////////////
        // Decompose and replace this double for loop with GEMV call

        cblas_dgemv(CblasRowMajor, CblasNoTrans, ann->hidden, ann->hidden+1, 1, w, ann->hidden+1, temp_i, 1, 0, sums, 1);
        for(j = 0; j < ann->hidden; ++j)
            o[j] = genann_act_hidden(ann, sums[j]);
        
        /*for (j = 0; j < ann->hidden; ++j) {
            for (k = 0; k < ann->hidden+1; ++k) {
                sums[j] += w[k + j*(ann->hidden+1)] * temp_i[k];
            }
            o[j] = genann_act_hidden(ann, sums[j]);
        }*/
        ////////////////////////////////////////////////////////////

        w += (n + 1) * m;
        o += m;
        i += m;
    }
    free(temp_i);
    free(sums);

    /////////////////////////////////
    // TODO 1 END               //
    /////////////////////////////////

    double const *ret = o;
    /* Figure output layer. */
    for (j = 0; j < ann->outputs; ++j) {
        double sum = *w++ * -1.0;
        for (k = 0; k < ann->hidden; ++k) {
            sum += *w++ * i[k];
        }
        *o++ = genann_act_output(ann, sum);
    }

    /* Sanity check that we used all weights and wrote all outputs. */
    assert(w - ann->weight == ann->total_weights);
    assert(o - ann->output == ann->total_neurons);

    return ret;
}


void genann_train(genann const *ann, double const *inputs, double const *desired_outputs, double learning_rate) {
    /* To begin with, we must run the network forward. */
    genann_run(ann, inputs);

    int h, j, k;


    /*
    TDT4200 comment:  This lonesome curly bracket starts a new scope.
                      That means no variables declared within will be
                      visible outside outside it (that is, o, d and t).
                      That also means the names can safely be re-used later.
    */
    {   /* First set the output layer deltas. */
        double const *o = ann->output + ann->inputs + ann->hidden * ann->hidden_layers; /* First output. */
        double *d = ann->delta + ann->hidden * ann->hidden_layers; /* First delta. */
        double const *t = desired_outputs; /* First desired output. */


        /* Set output layer deltas. */
        if (genann_act_output == genann_act_linear ||
                ann->activation_output == genann_act_linear) {
            for (j = 0; j < ann->outputs; ++j) {
                *d++ = *t++ - *o++;
            }
        } else {
            for (j = 0; j < ann->outputs; ++j) {
                *d++ = (*t - *o) * *o * (1.0 - *o);
                ++o; ++t;
            }
        }
    }

    /* Set hidden layer deltas, start on last layer and work backwards. */
    /* Note that loop is skipped in the case of hidden_layers == 0. */
    for (h = ann->hidden_layers - 1; h >= 0; --h) {

        /* Find first output and delta in this layer. */
        double const *o = ann->output + ann->inputs + (h * ann->hidden);
        double *d = ann->delta + (h * ann->hidden);

        /* Find first delta in following layer (which may be hidden or output). */
        double const * const dd = ann->delta + ((h+1) * ann->hidden);

        /* Find first weight in following layer (which may be hidden or output). */
        double const * const ww = ann->weight + ((ann->inputs+1) * ann->hidden) + ((ann->hidden+1) * ann->hidden * (h));


        /////////////////////////////////
        // TODO: 2 GEMV using BLAS   //
        /////////////////////////////////

        /*
        Function description:
        First, run an input through the network (see TODO 1).
        Then, starting at the output layer and running backwards, determine the
        deltas, the differences between the desired activation and actual
        activation of each neuron.

        Variables:
        ann: The artificial neural network, same as in TODO 1. \
             ann->delta: An area of memory as big as ann->output.
                        When function returns, stores difference between desired
                        output and actual output.
        h:  The index of the current layer. Starting at the output layer and
            going backwards.
        o:  The actual output of the current layer, produced by TODO 1 function.
        d:  The delta vector of the current layer, this is what we're calculating
        dd: The delta vector of the *previous* layer, at index (h+1).
        ww: The edge weights going from the previous layer and the current one.

        Note 1: We're playing fast and loose with the term "difference" here.
                It is not strictly a [desired]-[actual] calculation, but can be
                conceptually thought of as a difference.
        Note 2: We say "previous" layer meaning the layer involved in the
                previous calculation. Because we are going backwards, this is,
                strictly speaking, the "following" layer, which is what it is
                called in the source code comments.
        */

        // TODO 2.a: Define the m and n dimension of the delta matrix
        // Hint: Look at the double for loop
        int m = (h == ann->hidden_layers-1) ? ann->outputs : ann->hidden;
        int n = ann->hidden;

        //A temporary vector to store the propagated delta from the previuos layer.
        double* delta = calloc(ann->hidden, sizeof(double));

        // TODO 2.b: Decompose and implement GEMV BLAS call for the code
        // Hint: Think about how ww is offset from its original address.
        // You will need pointer arithmetic for the BLAS call

        
        cblas_dgemv(CblasRowMajor, CblasTrans, m, n, 1, ww+1, n, dd, 1, 0, delta, 1);
        // HOTSPOT Part 2 After: ~13%
        for(j = 0; j < ann->hidden; ++j)
            d[j] = o[j] * (1.0-o[j]) * delta[j];

        free(delta);

        /* // TODO 2.a: Define the m and n dimension of the delta matrix
        // Hint: Look at the double for loop
        int m = 0;
        int n = 0;

        //A temporary vector to store the propagated delta from the previuos layer.
        double* delta = calloc(ann->hidden, sizeof(double));

        // TODO 2.b: Decompose and implement GEMV BLAS call for the code
        // Hint: Think about how ww is offset from its original address.
        // You will need pointer arithmetic for the BLAS call

        // HOTSPOT Part 1: ~35%
        // HOTSPOT Part 1 after: ~43% 
        for (j = 0; j < ann->hidden; ++j) {
            //We iterate up to the value ann->outputs if we are on the output layer,
            //h == ann->hidden_layers-1, and to ann->hidden otherwise.
            for (k = 0; k < (h == ann->hidden_layers-1 ? ann->outputs : ann->hidden); ++k) {
                //Similarly to TODO 1, we add 1 to ann->hidden and j here as a
                //consequence of how bias for each neuron is stored.
                const int windex = k * (ann->hidden + 1) + (j + 1);
                //Propagate the deltas from the previous layer backwards
                //Using the weights between the layers and storing the result in "delta"
                delta[j] += dd[k] * ww[windex];
            }
            //Calculate the actual new deltas for this layer
            d[j] = o[j] * (1.0-o[j]) * delta[j];
        }

        free(delta); */

        /////////////////////////////////
        // TODO 2 END               //
        /////////////////////////////////
    }


    /* Train the outputs. */
    {
        /* Find first output delta. */
        double const *d = ann->delta + ann->hidden * ann->hidden_layers; /* First output delta. */

        /* Find first weight to first output delta. */
        double *w = ann->weight + (ann->hidden_layers
                ? ((ann->inputs+1) * ann->hidden + (ann->hidden+1) * ann->hidden * (ann->hidden_layers-1))
                : (0));

        /* Find first output in previous layer. */
        double const * const i = ann->output + (ann->hidden_layers
                ? (ann->inputs + (ann->hidden) * (ann->hidden_layers-1))
                : 0);

        /* Set output layer weights. */
        for (j = 0; j < ann->outputs; ++j) {
            *w++ += *d * learning_rate * -1.0;
            for (k = 1; k < (ann->hidden_layers ? ann->hidden : ann->inputs) + 1; ++k) {
                *w++ += *d * learning_rate * i[k-1];
            }

            ++d;
        }

        assert(w - ann->weight == ann->total_weights);
    }


    /* Train the hidden layers. */
    for (h = ann->hidden_layers - 1; h >= 0; --h) {

        /* Find first delta in this layer. */
        double const *d = ann->delta + (h * ann->hidden);

        /* Find first input to this layer. */
        double const *i = ann->output + (h
                ? (ann->inputs + ann->hidden * (h-1))
                : 0);

        /* Find first weight to this layer. */
        double *w = ann->weight + (h
                ? ((ann->inputs+1) * ann->hidden + (ann->hidden+1) * (ann->hidden) * (h-1))
                : 0);


    /////////////////////////////////
    // TODO: 3 (Optional) Optimize //
    /////////////////////////////////
        // HOTSPOT Part 1 after: ~30%
        // HOTSPOT Part 2 after: ~50%
        for (j = 0; j < ann->hidden; ++j) {
            *w++ += *d * learning_rate * -1.0;
            for (k = 1; k < (h == 0 ? ann->inputs : ann->hidden) + 1; ++k) {
                *w++ += *d * learning_rate * i[k-1];
            }
            ++d;
        }
    /////////////////////////////////
    // TODO 3 END               //
    /////////////////////////////////
    }

}


void genann_write(genann const *ann, FILE *out) {
    fprintf(out, "%d %d %d %d", ann->inputs, ann->hidden_layers, ann->hidden, ann->outputs);

    int i;
    for (i = 0; i < ann->total_weights; ++i) {
        fprintf(out, " %.20e", ann->weight[i]);
    }
}
