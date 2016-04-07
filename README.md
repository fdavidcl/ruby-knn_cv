# KnnCv

A native kNN leave-one-out technique implementation (oriented to feature selection) for Ruby based on the 'class' package for R.

## Installation

Add this line to your application's Gemfile:

```ruby
gem 'knn_cv'
```

And then execute:

    $ bundle

Or install it yourself as:

    $ gem install knn_cv

## Usage

This gem provides very basic functionality in a straight-forward manner. `Classifier` objects can be created in order to evaluate kNN several times:

```ruby
knn = KnnCv::Classifier.new k, data, Random.new
```

where `k` represents the number of neighbors to be considered, `data` must be an `Array` of instances, which in turn are numeric `Array`s. The third argument is a `Random` object that can be used to ensure reproducibility.

To evaluate the behavior of the classifier over certain features, use the `fitness_for` method:

```ruby
knn.fitness_for [0, 1, 0, 0, 1, 0, 1, 1, 1]
```

The argument must be a binary `Array` (or other type of object that converts to a binary `Array`, such as [`BitArray`](https://github.com/ingramj/bitarray)).

## Contributing

Bug reports and pull requests are welcome at https://github.com/fdavidcl/knn_cv.
