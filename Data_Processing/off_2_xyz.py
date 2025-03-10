def off_to_xyz(off_file, xyz_file):
    with open(off_file, 'r') as f:
        lines = f.readlines()

    if lines[0].strip() != "OFF":
        raise ValueError("Not a valid OFF file")

    parts = lines[1].strip().split()
    num_vertices = int(parts[0])
    
    # Read vertices
    vertices = []
    for i in range(2, 2 + num_vertices):
        vertex = lines[i].strip().split()
        # Ensure the vertex has 3 coordinates (x, y, z)
        if len(vertex) == 3:
            vertices.append(vertex)

    with open(xyz_file, 'w') as f:
        for vertex in vertices:
            f.write(" ".join(vertex) + "\n")
    
    print(f"{num_vertices} vertices converted from {off_file} to {xyz_file}")

off_to_xyz("ModelNet40/airplane/test/airplane_0628.off", "airplane_0628.xyz")