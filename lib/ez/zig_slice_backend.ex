defmodule Ez.ZigSliceBackend do
  @behaviour Nx.Tensor

  use Zig

  defstruct [:ref]

  alias Inspect.Algebra, as: IA

  defp size_of({_, size}), do: div(size, 8)

  defp count(%Nx.Tensor{shape: shape}) do
    shape
    |> Tuple.to_list
    |> Enum.reduce(1, &Kernel.*/2)
  end

  # GENERIC BROADCASTING OPERATIONS

  def op_name(op, {:s, size}) do
    String.to_existing_atom("#{op}_i#{size}")
  end
  def op_name(op, {class, size}) when class in [:u, :f] do
    String.to_existing_atom("#{op}_#{class}#{size}")
  end

  @concurrency_mode :dirty_cpu
  @operations ~w(add multiply)a
  @usable_types ~w(i64 i32 i16 i8 u64 u32 u16 u8 f64 f32)a

  for operation <- [:add, :multiply] do
    @impl true
    def unquote(operation)(a, b, c) do
      a_ref = allocate(count(a) * size_of(a.type))
      a_dim = Tuple.to_list(a.shape)

      b_ref = b.data.ref
      b_dim = Tuple.to_list(b.shape)

      c_ref = c.data.ref
      c_dim = Tuple.to_list(c.shape)

      op = op_name(unquote(operation), a.type)

      apply(__MODULE__, op, [a_ref, a_dim, b_ref, b_dim, c_ref, c_dim])
      %{a | data: %__MODULE__{ref: a_ref}}
    end

    for type <- @usable_types do
      ~z"""
      /// nif: #{operation}_#{type}/6 #{@concurrency_mode}
      fn #{operation}_#{type}(env: beam.env, dr: beam.term, dd: []usize, lr: beam.term, ld: []usize, rr: beam.term, rd: []usize) beam.term {
        return broadcast(#{type}, env, dr, dd, lr, ld, rr, rd, #{operation});
      }
      """
    end
  end

  # OPERATION IMPLEMENTATIONS

  ~Z"""
  fn add(left: anytype, right: anytype) @TypeOf(left) { return left + right; }
  fn multiply(left: anytype, right: anytype) @TypeOf(left) { return left * right; }
  """

  # BROADCASTING AND BROADCASTING TOOLS

  ~Z"""
  fn BinaryFn(comptime T: type) type { return fn (anytype, anytype) T;}

  // increments the tensor indices in a row-major fashion.  Note that
  // for column-major systems, the last index increments the fastest.
  fn increment_tensor_indices(indices: []usize, dims: [] const usize) void {
    var dim = dims.len - 1;
    while (dim >= 0) : (dim -= 1) {
      indices[dim] += 1;
      if (indices[dim] != dims[dim]) { break; }
      indices[dim] = 0;
      if (dim == 0) { break; }
    }
  }

  // finds an element in the tensor, indexed by the `indices` parameter.
  // this can't be a simple lookup since we might be broadcasting.
  // broadcasting is determined by if the source dimension is 1.
  // assumes that
  fn find(comptime T: type, src: []const T, src_dims: []const usize, indices: []const usize) T {
    var dim = src_dims.len - 1;
    var index: usize = 0;
    var multiplier: usize = 1;
    while (dim >= 0) : (dim -= 1) {
      var this_dim = src_dims[dim];
      if (this_dim != 1) {
        index += multiplier * indices[dim];
        multiplier *= src_dims[dim];
      }
      if (dim == 0) { break; }
    }
    return src[index];
  }

  // reinterprets a slice of u8 bytes into a slice of arbitrary types.
  // this function may fail if your type has alignment larger than that
  // of the beam term (currently u64).  So, no complex float64s or
  // quaternion float32s, please.
  fn reslice(comptime T: type, orig: []u8) []T {
    return @ptrCast(
      [*]T,
      @alignCast(
        @alignOf([*]T),
        orig.ptr))
      [0..(orig.len / @sizeOf(T))];
  }

  fn broadcast(comptime T: type,
               env: beam.env,
               dest_rsrc: beam.term,
               dest_dims: []usize,
               left_rsrc: beam.term,
               left_dims: []usize,
               right_rsrc: beam.term,
               right_dims: []usize,
               f:BinaryFn(T)) beam.term {

    var indices = beam.allocator.alloc(usize, dest_dims.len)
      catch return beam.raise_enomem(env);
    defer beam.allocator.free(indices);
    // zero out the indices.
    for (indices) |*i| {i.* = 0;}

    const dest_tensor_u8 = __resource__.fetch(tensor_slice, env, dest_rsrc)
      catch return beam.raise_resource_error(env);
    const left_tensor_u8 = __resource__.fetch(tensor_slice, env, left_rsrc)
      catch return beam.raise_resource_error(env);
    const right_tensor_u8 = __resource__.fetch(tensor_slice, env, right_rsrc)
      catch return beam.raise_resource_error(env);

    const dest_tensor = reslice(T, dest_tensor_u8);
    const left_tensor = reslice(T, left_tensor_u8);
    const right_tensor = reslice(T, right_tensor_u8);

    var left: T = undefined;
    var right: T = undefined;

    for (dest_tensor) |*v| {
      left = find(T, left_tensor, left_dims, indices);
      right = find(T, right_tensor, right_dims, indices);

      v.* = f(left, right);

      increment_tensor_indices(indices, dest_dims);
    }

    return beam.make_ok(env);
  }
  """

  # RESOURCES AND RESOURCE ACCESS

  ~Z"""
  /// in zigland, our tensor is represented as slice of u8.  This
  /// can be converted to arbitrary "other tensors" as needed.
  /// resource: tensor_slice definition
  const tensor_slice = []u8;

  /// resource: tensor_slice cleanup
  fn tensor_slice_cleanup(env: beam.env, tensor: *tensor_slice) void {
    beam.allocator.free(tensor.*);
  }

  /// nif: allocate_binary/2
  fn allocate_binary(env: beam.env, binary: beam.term, size: usize) beam.term {
    // retrieve the slice from the binary data.
    const source = beam.get_char_slice(env, binary) catch
      return beam.raise_function_clause_error(env);

    const tensor = beam.allocator.alloc(u8, size) catch
      return beam.raise_enomem(env);
    errdefer beam.allocator.free(tensor);

    // copy from source into the tensor
    std.mem.copy(u8, tensor, source);
    return __resource__.create(tensor_slice, env, tensor) catch
      return beam.raise_enomem(env);
  }

  /// nif: allocate/1
  fn allocate(env: beam.env, size: usize) beam.term {
    const tensor = beam.allocator.alloc(u8, size) catch
      return beam.raise_enomem(env);
    errdefer beam.allocator.free(tensor);

    return __resource__.create(tensor_slice, env, tensor) catch
      return beam.raise_enomem(env);
  }

  /// nif: read_impl/1
  fn read_impl(env: beam.env, source: beam.term) beam.term {
    var tensor = __resource__.fetch(tensor_slice, env, source)
      catch return beam.raise_resource_error(env);

    return beam.make_slice(env, tensor);
  }
  """

  @impl true
  def inspect(tensor, opts) do
    IA.concat(["<", IA.to_doc(tensor.data.ref, opts), ">"])
  end

  @impl true
  def from_binary(tensor, data, _opts \\ []) do
    ref = allocate_binary(data, count(tensor) * size_of(tensor.type))
    %{tensor | data: %__MODULE__{ref: ref}}
  end

  @impl true
  def to_binary(%{data: %__MODULE__{ref: ref}}, _opts \\ []) do
    read_impl(ref)
  end

end
