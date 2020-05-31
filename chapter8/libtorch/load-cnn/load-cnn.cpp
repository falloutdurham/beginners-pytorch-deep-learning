#include <torch/script.h> 
#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
  
  torch::jit::script::Module  module = torch::jit::load("cnnnet");

  std::cout << "model loaded ok\n";

  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::rand({1, 3, 224, 224}));

  at::Tensor output = module.forward(inputs).toTensor();

  std::cout << output << '\n';
}
