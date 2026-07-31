async function updatePrediction(){

    const response = await fetch("/prediction");

    const data = await response.json();

    document.getElementById("letter").innerText = data.letter;
    document.getElementById("confidence").innerText = data.confidence + "%";
    document.getElementById("word").innerText = data.word;
    document.getElementById("sentence").innerText = data.sentence;

}
async function addLetter() {
    await fetch("/add_letter", {
        method: "POST"
    });
}

async function addWord() {

    await fetch("/add_word", {
        method: "POST"
    });

}
async function clearText(){

    await fetch("/clear",{
        method:"POST"
    });

}
async function speak(){

    await fetch("/speak",{
        method:"POST"
    });

}
async function backspace(){

    await fetch("/backspace",{
        method:"POST"
    });

}
async function exitApp() {

    const response = await fetch("/exit", {
        method: "POST"
    });

    if (response.ok) {
        document.body.innerHTML = `
            <div style="
                height:100vh;
                display:flex;
                justify-content:center;
                align-items:center;
                background:#111827;
                color:white;
                font-size:42px;
                font-family:Arial;">
                👋 Thank you for using GestureSense AI
            </div>
        `;
    }
}
setInterval(updatePrediction,300);