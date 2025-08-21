# pip install torchviz graphviz netron-export
import torch
from torchviz import make_dot
from Model import Net

# Create your model
model = Net()

# Create dummy input
x = torch.rand(1, 3, 224, 224)

# Forward pass
output = model(x)
print(f"Model output shape: {output.shape}")

# Save model as JIT for netron
jit_model_path = "architecture_diagram_temp.pth"  # Archivo temporal
traced_model = torch.jit.trace(model, x)
torch.jit.save(traced_model, jit_model_path)

# Export with netron (if you have netron_export)
try:
    from netron_export import export_graph
    svg_path = 'architectureDiagrams/resnet_inversion_netron.png'
    export_graph(jit_model_path, [svg_path], False, 8483, 50000)
    print(f"Netron SVG saved to: {svg_path}")
except ImportError:
    print("netron_export not available, skipping...")

# # Create visualization with torchviz
# svg_path = 'models/resnet_inversion_viz'
# dot = make_dot(output, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
# dot.render(svg_path, directory='.', format='svg')
# print(f"Torchviz SVG saved to: {svg_path}.svg")

# # Also create PNG version
# dot.render(svg_path, directory='.', format='png')
# print(f"Torchviz PNG saved to: {svg_path}.png")