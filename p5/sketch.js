const drawable = new Ship()

function setup() {
    createCanvas(800, 800)
    drawable.setup()
}


function draw() {
    // if (mouseIsPressed) {
    //     fill(0);
    // } else {
    //     fill(255);
    // }
    // ellipse(mouseX, mouseY, 80, 80)
    drawable.draw()
    // translation = 0
    
    // plano de linha do alto
    line(0 + translation,400 + translation,40 + translation,500 + translation)
    line(40 + translation,500 + translation,500 + translation,500 + translation)
    line(500 + translation,500 + translation,540 + translation,400 + translation)
    line(540 + translation,400 + translation, 0  + translation, 400 + translation)
    
    
   
    // line(0,400,40,500)
    // line(40,500,500,500)
    // line(500,500,540,400)
    // line(540,400, 0 , 400)
    
    

}