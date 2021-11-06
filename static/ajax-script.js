let avatar;
$(document).on('submit', '#animate-form', function (e) {
    console.log('hello');
    e.preventDefault();
    $.ajax({
        type: 'POST',
        url: 'requests',
        data: {
            animate: 1,
            avatar: avatar
        },
        success: function (result) {
        }
    })
});

const buttons = document.querySelectorAll('.button');

buttons.forEach(elem => {
    elem.addEventListener("click", (e) => {
        //document.querySelector(".streaming-el:nth-child(2) img").src = elem.childNodes[1].attributes.src.value;
        avatar = elem.childNodes[1].attributes.src.value;
        console.log(avatar)
    })

})