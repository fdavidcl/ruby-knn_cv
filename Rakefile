require "bundler/gem_tasks"
require "rake/testtask"

Rake::TestTask.new(:test) do |t|
  t.libs << "test"
  t.libs << "lib"
  t.test_files = FileList['test/**/*_test.rb']
end

task :default => :spec

require 'rake/extensiontask'
spec = Gem::Specification.load('feature-selection.gemspec')
Rake::ExtensionTask.new('c_knn', spec)
