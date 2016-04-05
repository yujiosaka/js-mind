const gulp = require('gulp')
const gutil = require('gulp-util')
const babel = require('gulp-babel')
const plumber = require('gulp-plumber')
const dirname = require('path').dirname

const srcdir  = __dirname + '/src'
const destdir = __dirname + '/dist'

/**
 * 1. transpile all babel files
 * 2. watch src dir
 * 3. transpile on changed
 */
gulp.task('watch', x => {

  gulp.start('babel:all')

  gulp.watch('src/**/*.js', (info) => {

    const src = info.path
    const relpath = src.slice(srcdir.length + 1)
    const dest = destdir + '/' + dirname(relpath)

    gutil.log(`[${info.type}]: ${relpath}`)
    compileBabel(src, dest)
      .on('end', x => {
        gutil.log(`compilation finished: ${dest}`)
    })
  })
})

/**
 * transpile all babel files
 */
gulp.task('babel:all', x => {
  compileBabel('src/**/*.js', 'dist')
    .on('end', x => {
      gutil.log('compilation finished: all js files.')
  })
})


/**
 * transipile babel src to dest
 */
function compileBabel(src, dest) {
  return gulp.src(src)
    .pipe(plumber())
    .pipe(babel())
    .pipe(gulp.dest(dest))
}
