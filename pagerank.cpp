#include <iostream>
#include <string>
#include <ctime>
#include <fstream>
#include <cmath>

using namespace std;

void save(double *pagerank, unsigned N)
{
    ofstream off("pagerank.out");
    double pmin = 1.0, pmax = 0.0;
    unsigned imin = 0, imax = 0;
    for (unsigned i = 0; i < N; i++)
    {
        off << pagerank[i] << endl;

        if (pagerank[i] > pmax)
        {
            pmax = pagerank[i];
            imax = i;
        }
        if (pagerank[i] < pmin)
        {
            pmin = pagerank[i];
            imin = i;
        }
    }
    off.close();

    cout << "Min: " << pmin << ", ind = " << imin << endl;
    cout << "Max: " << pmax << ", ind = " << imax << endl;
}

void pagerank(Graph *graph, double alpha = 0.85, unsigned maxiter = 100)
{
    const double tolerance = 1e-8;

    unsigned nvert = graph->vertex_count;
    double scale = 1.0 / nvert;
    double *pagerank = new double[nvert];
    double *last = new double[nvert];
    double teleport = (1.0 - alpha) / nvert;

    #pragma omp parallel for
    for (unsigned i = 0; i < nvert; i++)
    {
        pagerank[i] = scale;
        last[i] = 0.0;
    }

    unsigned iter = 0;
    while (iter++ < maxiter)
    {
        #pragma omp parallel for
        for (unsigned i = 0; i < nvert; i++)
        {
            last[i] = pagerank[i];
            pagerank[i] = 0.0;
        }

        // dangling nodes that have no outgoing edges
        double zsum = 0.0;
        #pragma omp parallel for reduction(+:zsum)
        for (unsigned i = 0; i < nvert; i++)
            if (graph->count_edges(i) == 0)
                zsum += last[i];
        // distribute their sum to all nodes
        double nolinks = alpha * zsum / nvert;

        #pragma omp parallel for
        for (vertex_id id = 0; id < nvert; id++)
        {
            double update = alpha * last[id] / graph->count_edges(id);
            for (Graph::iterator e = graph->iterate_outgoing_edges(id); !e.end(); e++)
            {
                #pragma omp atomic
                pagerank[(*e).v2] += update;
            }
            #pragma omp atomic
            pagerank[id] += teleport + nolinks;

            if (id % 10000000 == 0)
                cout << ".";
        }
        cout << endl;

        // sum the pagerank
        double sum = 0.0;
        #pragma omp parallel for reduction(+:sum)
        for (unsigned i = 0; i < nvert; i++)
            sum += pagerank[i];

        // normalize to valid probabilities i.e. 0 to 1
        sum = 1.0 / sum;
        #pragma omp parallel for 
        for (unsigned i = 0; i < nvert; i++)
            pagerank[i] *= sum;

        // sum up the error
        double err = 0.0;
        #pragma omp parallel for reduction(+:err)
        for (unsigned i = 0; i < nvert; i++)
            err += fabs(pagerank[i] - last[i]);

        // error is small enough, we are done
        if (err < tolerance)
            break;

        cout << "Iteration " << iter << endl;
        cout << "Error: " << err << endl;
    }

    save(pagerank, nvert);

    delete[] last;
    delete[] pagerank;
}

int main()
{
#ifndef DEBUG
    Graph *graph = get_debug_graph();
#else
    Graph *graph = get_normal_graph();
#endif

    pagerank(graph);

    delete graph;
    return 0;
}

