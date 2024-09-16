#include<bits/stdc++.h>
using namespace std;

struct Weight{
    vector<double> w; // weights

    Weight(int size){
        w = vector<double>(size , 0.0);
    }
};
struct Bias{
    double b; // bias

    Bias(){
        b = 0.0;
    }
};
struct Layer {
    vector<Weight> w; // weights
    vector<Bias> b;   // bias
    
    Layer(int size, int input_size) {
        for (int i = 0; i < size; ++i) {
            w.push_back(Weight(input_size));
            b.push_back(Bias());
        }
    }
};

class Model{
    public:
        vector<Layer> layers;
        
        void dense(int size , int input_size){
            layers.push_back(Layer(size , input_size));
        }

        void train(vector<vector<int>> x){ // vector of all flattened images ("features")
            for(int i =0; i < x.size(); ++i){
                forwardPropagation(x[i]);
                backPropagation(x[i]);
            }
        }
    private:
        void forwardPropagation(vector<int> x){
            for(auto& layer: layers){
                int size = layer.w.size();
                for(int i = 0; i < size; ++i){ // a single neuron has w[i] weights and b[i] bias
                    
                }
            }
        }
        void backPropagation(vector<int> x){

        }
};

int main(){
    Model model;
    model.dense(32 , 28*28);
    model.dense(16 , 28*28);
    model.dense(4 , 28*28);
    cout << model.layers.size() << '\n';
    cout << model.layers[0].w.size() << '\n';
    return 0;
}