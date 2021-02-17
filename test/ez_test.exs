defmodule EzTest do
  use ExUnit.Case
  doctest Ez

  test "greets the world" do
    assert Ez.hello() == :world
  end
end
