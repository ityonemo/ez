defmodule EzTest do
  use ExUnit.Case, async: true

  defp commute(a, b, fun) do
    fun.(a, b)
    fun.(b, a)
  end

  def tensor(t, opts \\ []) do
    t
    |> Nx.tensor(opts)
    |> Nx.backend_transfer(Ez.ZigSliceBackend)
  end

  describe "addition" do
    test "with i64" do
      commute(tensor([[1, 2], [3, 4]]), tensor([[5, 6], [7, 8]]), fn a, b ->
        t = Nx.add(a, b)

        assert Nx.to_binary(t) ==
                 <<6::64-native, 8::64-native, 10::64-native, 12::64-native>>

        assert Nx.shape(t) == {2, 2}
      end)
    end

    test "with f32" do
      commute(tensor([[1.0, 2.0], [3.0, 4.0]], type: {:f, 32}), tensor([[5.0, 6.0], [7.0, 8.0]], type: {:f, 32}), fn a, b ->
        t = Nx.add(a, b)

        assert Nx.to_binary(t) ==
                 <<6.0::32-float-native, 8.0::32-float-native, 10.0::32-float-native, 12.0::32-float-native>>

        assert Nx.shape(t) == {2, 2}
      end)
    end
  end

  describe "multiplication" do
    test "with i64" do
      commute(tensor([[1, 2], [3, 4]]), tensor([[5, 6], [7, 8]]), fn a, b ->
        t = Nx.multiply(a, b)

        assert Nx.to_binary(t) ==
                 <<5::64-native, 12::64-native, 21::64-native, 32::64-native>>

        assert Nx.shape(t) == {2, 2}
      end)
    end

    test "with f32" do
      commute(tensor([[1.0, 2.0], [3.0, 4.0]], type: {:f, 32}), tensor([[5.0, 6.0], [7.0, 8.0]], type: {:f, 32}), fn a, b ->
        t = Nx.multiply(a, b)

        assert Nx.to_binary(t) ==
                 <<5.0::32-float-native, 12.0::32-float-native, 21.0::32-float-native, 32.0::32-float-native>>

        assert Nx.shape(t) == {2, 2}
      end)
    end
  end

  describe "binary broadcast" do
    test "{2, 1} + {1, 2}" do
      commute(tensor([[1], [2]]), tensor([[10, 20]]), fn a, b ->
        t = Nx.add(a, b)

        assert Nx.to_binary(t) ==
                 <<11::64-native, 21::64-native, 12::64-native, 22::64-native>>

        assert Nx.shape(t) == {2, 2}
      end)
    end

    test "{2} + {2, 2}" do
      commute(tensor([1, 2]), tensor([[1, 2], [3, 4]]), fn a, b ->
        t = Nx.add(a, b)

        assert Nx.to_binary(t) ==
                 <<2::64-native, 4::64-native, 4::64-native, 6::64-native>>

        assert Nx.shape(t) == {2, 2}
      end)
    end
  end
end
