class Drawable {
    constructor () {
        console.log("Instanciando objeto desenhável")
    }
    setup() {
        throw new Error("Not implemented error")
    }

    draw() {
        throw new Error("Not implemented error")
    }
}


class Ship extends Drawable{
    setup() {
        return 0;
    }

    draw() {
        translation = 0
    
        // plano de linha do alto
        line(0 + translation,400 + translation,40 + translation,500 + translation)
        line(40 + translation,500 + translation,500 + translation,500 + translation)
        line(500 + translation,500 + translation,540 + translation,400 + translation)
        line(540 + translation,400 + translation, 0  + translation, 400 + translation)
    }
}