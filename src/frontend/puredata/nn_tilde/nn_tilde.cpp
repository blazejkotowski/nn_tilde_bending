#include "../../../backend/backend.h"
#include "../../maxmsp/shared/circular_buffer.h"
#include "m_pd.h"
#include <memory>
#include <mutex>
#include <string>
#include <vector>

static t_class *nn_tilde_class;

#ifndef VERSION
#define VERSION "UNDEFINED"
#endif

unsigned power_ceil(unsigned x) {
  if (x <= 1)
    return 1;
  int power = 2;
  x--;
  while (x >>= 1)
    power <<= 1;
  return power;
}

// CLASS LIKE INITIALISATION
typedef struct _nn_tilde {
  t_object x_obj;
  t_sample f;

  t_outlet *m_outlet; // Outlet for registered attributes

  int m_enabled;
  // BACKEND RELATED MEMBERS
  std::unique_ptr<Backend> m_model;
  std::vector<std::string> settable_attributes;
  t_symbol *m_method, *m_path;
  std::unique_ptr<std::thread> m_compute_thread;

  // BUFFER RELATED MEMBERS
  int m_head, m_in_dim, m_in_ratio, m_out_dim, m_out_ratio, m_buffer_size;

  std::unique_ptr<circular_buffer<float, float>[]> m_in_buffer;
  std::unique_ptr<circular_buffer<float, float>[]> m_out_buffer;
  std::vector<std::unique_ptr<float[]>> m_in_model, m_out_model;

  bool m_use_thread;

  // DSP RELATED MEMBERS
  int m_dsp_vec_size;
  std::vector<float *> m_dsp_in_vec;
  std::vector<float *> m_dsp_out_vec;

} t_nn_tilde;

void model_perform(t_nn_tilde *nn_instance) {
  std::vector<float *> in_model, out_model;

  for (int c(0); c < nn_instance->m_in_dim; c++)
    in_model.push_back(nn_instance->m_in_model[c].get());
  for (int c(0); c < nn_instance->m_out_dim; c++)
    out_model.push_back(nn_instance->m_out_model[c].get());

  nn_instance->m_model->perform(in_model, out_model, nn_instance->m_buffer_size,
                                nn_instance->m_method->s_name, 1);
}

// DSP CALL
t_int *nn_tilde_perform(t_int *w) {
  t_nn_tilde *x = (t_nn_tilde *)(w[1]);

  if (!x->m_model->is_loaded() || !x->m_enabled) {
    for (int c(0); c < x->m_out_dim; c++) {
      for (int i(0); i < x->m_dsp_vec_size; i++) {
        x->m_dsp_out_vec[c][i] = 0;
      }
    }
    return w + 2;
  }

  // COPY INPUT TO CIRCULAR BUFFER
  for (int c(0); c < x->m_in_dim; c++) {
    x->m_in_buffer[c].put(x->m_dsp_in_vec[c], x->m_dsp_vec_size);
  }

  if (x->m_in_buffer[0].full()) { // BUFFER IS FULL
    // IF USE THREAD, CHECK THAT COMPUTATION IS OVER
    if (x->m_compute_thread && x->m_use_thread) {
      x->m_compute_thread->join();
    }

    // TRANSFER MEMORY BETWEEN INPUT CIRCULAR BUFFER AND MODEL BUFFER
    for (int c(0); c < x->m_in_dim; c++)
      x->m_in_buffer[c].get(x->m_in_model[c].get(), x->m_buffer_size);

    if (!x->m_use_thread) // PROCESS DATA RIGHT NOW
      model_perform(x);

    // TRANSFER MEMORY BETWEEN OUTPUT CIRCULAR BUFFER AND MODEL BUFFER
    for (int c(0); c < x->m_out_dim; c++)
      x->m_out_buffer[c].put(x->m_out_model[c].get(), x->m_buffer_size);

    if (x->m_use_thread) // PROCESS DATA LATER
      x->m_compute_thread = std::make_unique<std::thread>(model_perform, x);
  }

  // COPY CIRCULAR BUFFER TO OUTPUT
  for (int c(0); c < x->m_out_dim; c++) {
    x->m_out_buffer[c].get(x->m_dsp_out_vec[c], x->m_dsp_vec_size);
  }

  return w + 2;
}

void nn_tilde_dsp(t_nn_tilde *x, t_signal **sp) {
  x->m_dsp_vec_size = sp[0]->s_n;
  x->m_dsp_in_vec.clear();
  x->m_dsp_out_vec.clear();

  for (int i(0); i < x->m_in_dim; i++) {
    x->m_dsp_in_vec.push_back(sp[i]->s_vec);
  }
  for (int i(x->m_in_dim); i < x->m_in_dim + x->m_out_dim; i++) {
    x->m_dsp_out_vec.push_back(sp[i]->s_vec);
  }
  dsp_add(nn_tilde_perform, 1, x);
}

void nn_tilde_free(t_nn_tilde *x) {
  if (x->m_compute_thread) {
    x->m_compute_thread->join();
  }
}

void *nn_tilde_new(t_symbol *s, int argc, t_atom *argv) {
  std::cout << "nn_tilde_new: entering function" << std::endl;

  t_nn_tilde *x = (t_nn_tilde *)pd_new(nn_tilde_class);

  x->m_model = std::make_unique<Backend>();
  x->m_head = 0;
  x->m_compute_thread = nullptr;
  x->m_in_dim = 1;
  x->m_in_ratio = 1;
  x->m_out_dim = 1;
  x->m_out_ratio = 1;
  x->m_buffer_size = 4096;
  x->m_method = gensym("forward");
  x->m_enabled = 1;
  x->m_use_thread = true;

  // CHECK ARGUMENTS
  if (!argc) {
    return (void *)x;
  }
  if (argc > 0) {
    x->m_path = atom_gensym(argv);
  }
  if (argc > 1) {
    x->m_method = atom_gensym(argv + 1);
  }
  if (argc > 2) {
    x->m_buffer_size = atom_getint(argv + 2);
  }

  // SEARCH FOR FILE
  if (!sys_isabsolutepath(x->m_path->s_name)) {
    char dirname[MAXPDSTRING], *dummy;
    auto fd = open_via_path("", x->m_path->s_name, "", dirname, &dummy,
                            MAXPDSTRING, 1);
    std::string found_path;
    found_path += dirname;
    found_path += "/";
    found_path += x->m_path->s_name;
    x->m_path = gensym(found_path.c_str());
  }


  std::cout << "nn_tilde_new: gonna try to load the model" << std::endl;
  // TRY TO LOAD MODEL
  if (x->m_model->load(x->m_path->s_name)) {
    post("error during loading");
    return (void *)x;
  } else {
    // cout << "successfully loaded model" << endl;
  }
  std::cout << "nn_tilde_new: loaded the model" << std::endl;

  std::cout << "nn_tilde_new: setting model's method params" << std::endl;
  // GET MODEL'S METHOD PARAMETERS
  auto params = x->m_model->get_method_params(x->m_method->s_name);
  x->settable_attributes = x->m_model->get_settable_attributes();

  if (!params.size()) {
    post("method not found, using forward instead");
    x->m_method = gensym("forward");
    params = x->m_model->get_method_params(x->m_method->s_name);
  }

  std::cout << "nn_tilde_new: setting in and out dimensions" << std::endl;

  std::cout << "nn_tilde_new: params.size() = " << params.size() << std::endl;
  // Print out the params to stdout with their indexes
  for (int i = 0; i < params.size(); i++) {
    std::cout << "params[" << i << "] = " << params[i] << std::endl;
  }
  x->m_in_dim = params[0];
  x->m_in_ratio = params[1];
  x->m_out_dim = params[2];
  x->m_out_ratio = params[3];

  auto higher_ratio = x->m_model->get_higher_ratio();

  if (!x->m_buffer_size) {
    // NO THREAD MODE
    x->m_use_thread = false;
    x->m_buffer_size = higher_ratio;
  } else if (x->m_buffer_size < higher_ratio) {
    x->m_buffer_size = higher_ratio;
    std::string err_message = "buffer size too small, switching to ";
    err_message += std::to_string(higher_ratio);
    post(err_message.c_str());
  } else {
    x->m_buffer_size = power_ceil(x->m_buffer_size);
  }

  std::cout << "nn_tilde_new: creating inlets and outlets" << std::endl;
  // CREATE INLETS, OUTLETS and BUFFERS
  x->m_in_buffer =
      std::make_unique<circular_buffer<float, float>[]>(x->m_in_dim);
  for (int i(0); i < x->m_in_dim; i++) {
    if (i < x->m_in_dim - 1)
      inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_signal, &s_signal);
    x->m_in_buffer[i].initialize(x->m_buffer_size);
    x->m_in_model.push_back(std::make_unique<float[]>(x->m_buffer_size));
  }

  x->m_out_buffer =
      std::make_unique<circular_buffer<float, float>[]>(x->m_out_dim);
  for (int i(0); i < x->m_out_dim; i++) {
    outlet_new(&x->x_obj, &s_signal);
    x->m_out_buffer[i].initialize(x->m_buffer_size);
    x->m_out_model.push_back(std::make_unique<float[]>(x->m_buffer_size));
  }

  // Add registered attribute outlet here
  x->m_outlet = outlet_new(&x->x_obj, &s_symbol); // Create extra outlet

  return (void *)x;
}

void nn_tilde_enable(t_nn_tilde *x, t_floatarg arg) {
  x->m_enabled = int(arg);
}
void nn_tilde_reload(t_nn_tilde *x) {
   x->m_model->reload();
}

void nn_tilde_set(t_nn_tilde *x, t_symbol *s, int argc, t_atom *argv) {
  std::cout << "nn_tilde_set: entering function" << std::endl;
  if (argc < 2) {
    post("set needs at least 2 arguments [set argname argval1 ...)");
    return;
  }
  std::vector<std::string> attribute_args;

  auto argname = argv[0].a_w.w_symbol->s_name;
  std::string argname_str = argname;
  std::cout << "nn_tilde_set: argname = " << argname << std::endl;

  if (!std::count(x->settable_attributes.begin(), x->settable_attributes.end(),
                  argname_str)) {
    post("argument name not settable in current model");
    return;
  }

  for (int i(1); i < argc; i++) {
    if (argv[i].a_type == A_SYMBOL) {
      std::cout << "nn_tilde_set: argv[i].a_w.w_symbol->s_name = " << argv[i].a_w.w_symbol->s_name << std::endl;
      attribute_args.push_back(argv[i].a_w.w_symbol->s_name);
    } else if (argv[i].a_type == A_FLOAT) {
      std::cout << "nn_tilde_set: argv[i].a_w.w_float = " << argv[i].a_w.w_float << std::endl;
      attribute_args.push_back(std::to_string(argv[i].a_w.w_float));
    }
  }
  try {
    std::cout << "nn_tilde_set: setting attribute '" << argname << "' to " << attribute_args << std::endl;
    x->m_model->set_attribute(argname, attribute_args);
  } catch (const std::exception &e) {
    post(e.what());
  }
}

void nn_tilde_get(t_nn_tilde *x, t_symbol *s) {
  /* Returns the attribute as symbol */
  std::cout << "nn_tilde_get: entering function" << std::endl;
  std::cout << "nn_tilde_get: getting attribute '" << s->s_name << "'" << std::endl;
  std::string attribute_value = x->m_model->get_attribute_as_string(s->s_name);
  std::cout << "nn_tilde_get: attribute value: " << attribute_value << std::endl;
  t_atom out_atom;
  SETSYMBOL(&out_atom, gensym(attribute_value.c_str()));
  outlet_anything(x->m_outlet, gensym("attribute"), 1, &out_atom);
}

// Get model layers and output through the designated outlet
void nn_tilde_layers(t_nn_tilde *x) {
  std::cout << "nn_tilde_layers: entering function" << std::endl;
  std::vector<std::string> layers = x->m_model->get_available_layers();
  std::cout << "nn_tilde_layers: layers.size() = " << layers.size() << std::endl;
  // for (int i = 0; i < layers.size(); i++) {
  //   std::cout << "nn_tilde_layers: layers[" << i << "] = " << layers[i] << std::endl;
  // }

  t_atom out_atom;
  t_atom *out_atoms = new t_atom[layers.size()];
  for (int i = 0; i < layers.size(); i++) {
    SETSYMBOL(&out_atoms[i], gensym(layers[i].c_str()));
  }

  outlet_anything(x->m_outlet, gensym("layers"), layers.size(), out_atoms);
  delete[] out_atoms;

}

// Get layer weights and output through the designated outlet
void nn_tilde_get_weights(t_nn_tilde *x, t_symbol *s) {
  std::cout << "nn_tilde_get_weights: entering function" << std::endl;
  std::string layer_name = s->s_name;
  std::cout << "nn_tilde_set_weights: layer_name = " << layer_name << std::endl;

  std::vector<float> layer_weights = x->m_model->get_layer_weights(layer_name);
  std::cout << "nn_tilde_get_weights: layer_weights.size() = " << layer_weights.size() << std::endl;

  t_atom out_atom;
  t_atom *out_atoms = new t_atom[layer_weights.size()];
  for (int i = 0; i < layer_weights.size(); i++) {
    SETFLOAT(&out_atoms[i], layer_weights[i]);
  }

  outlet_anything(x->m_outlet, gensym("layer"), layer_weights.size(), out_atoms);
  delete[] out_atoms;
}

// Set layer weights
void nn_tilde_set_weights(t_nn_tilde *x, t_symbol *s, int argc, t_atom *argv) {
  std::cout << "nn_tilde_set_weights: entering function" << std::endl;

  if (argv[0].a_type != A_SYMBOL) {
    post("set_weights: first argument must be a layer name");
    return;
  }

  std::string layer_name = atom_getsymbol(argv)->s_name;
  std::vector<float> layer_weights;

  std::cout << "nn_tilde_set_weights: layer_name = " << layer_name << std::endl;
  // std::cout << "nn_tilde_set_weights: argv[0] = " << argv[0].a_w.w_float << std::endl;

  for (int i = 0; i < argc; i++) {
    if (argv[i].a_type == A_FLOAT) {
      layer_weights.push_back(argv[i].a_w.w_float);
    }
  }
  std::cout << "nn_tilde_set_weights: layer_weights.size() = " << layer_weights.size() << std::endl;
  x->m_model->set_layer_weights(layer_name, layer_weights);
}

void startup_message() {
  std::string startmessage = "nn~ - ";
  startmessage += VERSION;
  startmessage += " - ";
  startmessage += "torch ";
  startmessage += TORCH_VERSION;
  startmessage += " - 2023 - Antoine Caillon";
  startmessage += " / 2024 - mod by Błażej Kotowski";
  post(startmessage.c_str());
}

extern "C" {
#ifdef _WIN32
void __declspec(dllexport) nn_tilde_setup(void) {
#else
void nn_tilde_setup(void) {
#endif
  startup_message();
  nn_tilde_class = class_new(gensym("nn~"), (t_newmethod)nn_tilde_new, 0,
                             sizeof(t_nn_tilde), CLASS_DEFAULT, A_GIMME, 0);

  class_addmethod(nn_tilde_class, (t_method)nn_tilde_dsp, gensym("dsp"), A_CANT,
                  0);
  class_addmethod(nn_tilde_class, (t_method)nn_tilde_enable, gensym("enable"),
                  A_DEFFLOAT, A_NULL);
  class_addmethod(nn_tilde_class, (t_method)nn_tilde_reload, gensym("reload"),
                  A_NULL);
  class_addmethod(nn_tilde_class, (t_method)nn_tilde_set, gensym("set"),
                  A_GIMME, A_NULL);
  // add method for retrieving a registered attribute
  class_addmethod(nn_tilde_class, (t_method)nn_tilde_get, gensym("get"),
                  A_SYMBOL, A_NULL);
  class_addmethod(nn_tilde_class, (t_method)nn_tilde_layers, gensym("layers"),
                  A_NULL);
  class_addmethod(nn_tilde_class, (t_method)nn_tilde_get_weights, gensym("get_weights"),
                  A_SYMBOL, A_NULL);
  class_addmethod(nn_tilde_class, (t_method)nn_tilde_set_weights, gensym("set_weights"),
                  A_GIMME, A_NULL);

  CLASS_MAINSIGNALIN(nn_tilde_class, t_nn_tilde, f);
}
}
