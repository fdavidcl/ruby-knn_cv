# coding: utf-8
lib = File.expand_path('../lib', __FILE__)
$LOAD_PATH.unshift(lib) unless $LOAD_PATH.include?(lib)
require 'knn_cv/version'

Gem::Specification.new do |spec|
  spec.name          = "knn_cv"
  spec.version       = KnnCv::VERSION
  spec.authors       = ["David Charte"]
  spec.email         = ["fdavidcl@outlook.com"]

  spec.summary       = %q{A native kNN leave-one-out technique implementation}
  spec.description   = %q{A native kNN leave-one-out technique implementation for Ruby based on the 'class' package for R.}
  spec.homepage      = "https://github.com/fdavidcl/ruby-knn_cv"
  spec.licenses      = ["GPL-3.0+"]

  # Prevent pushing this gem to RubyGems.org by setting 'allowed_push_host', or
  # delete this section to allow pushing this gem to any host.
  # if spec.respond_to?(:metadata)
  #   spec.metadata['allowed_push_host'] = "TODO: Set to 'http://mygemserver.com'"
  # else
  #   raise "RubyGems 2.0 or newer is required to protect against public gem pushes."
  # end

  spec.files         = `git ls-files -z`.split("\x0").reject { |f| f.match(%r{^(test|spec|features)/}) }
  spec.bindir        = "exe"
  spec.executables   = spec.files.grep(%r{^exe/}) { |f| File.basename(f) }
  spec.require_paths = ["lib"]
  spec.extensions   << "ext/c_knn/extconf.rb"

  spec.add_development_dependency "bundler", "~> 1.11"
  spec.add_development_dependency "rake", "~> 10.0"
  spec.add_development_dependency "minitest", "~> 5.0"
  spec.add_development_dependency "rake-compiler"
end
